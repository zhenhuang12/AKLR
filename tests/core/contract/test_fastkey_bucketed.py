# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import dataclasses
import sys
import types

import pytest
import torch

from aklr.core.contract.fastkey import FastKeyCodec
from aklr.core.contract.spec import AxisSpec
from aklr.core.runtime.dispatch import DispatchTable, FallbackReason
from tests.op_case import OpCase


class _Runnable:
    """A kernel whose module declares RANGE_SAFE as given (``_dispatcher._is_range_safe``)."""

    def __init__(self, monkeypatch, range_safe):
        name = f"tests_range_safe_{range_safe}"
        module = types.ModuleType(name)
        module.RANGE_SAFE = range_safe
        monkeypatch.setitem(sys.modules, name, module)
        self.metadata = types.SimpleNamespace(misc={"module_name": name})


def _point(case, n_tokens):
    return case.point.model_copy(update={"axes": {**case.point.axes, "n_tokens": n_tokens}})


def _key(codec, case, n_tokens):
    return codec.match_key(codec.to_fast_key(_point(case, n_tokens)))


def _schema(case, measured):
    """fused_moe's case narrowed to a schema measured at ``measured``: n_tokens -> speedup, None for a failed
    accuracy."""
    workloads = [
        case.contract.workload_for(_point(case, n), {}).model_copy(
            update={"accuracy": "passed", "speedup": s} if s is not None else {"accuracy": "failed"}
        )
        for n, s in sorted(measured.items())
    ]
    return case.contract.narrow(case.point, workloads=[w.model_dump(mode="json", exclude_none=True) for w in workloads],
                                kernel={"language": "triton", "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
                                        "entry_point": "kernel.py::run", "sources": []})  # fmt: skip


def _table(monkeypatch, case, *served):
    """The table of ``served``'s (schema, runnable) pairs at AKLR_MIN_SPEEDUP 1.0, whose lookup takes a Point as the
    call (its key the Point's), outside any graph capture."""
    # no device needed for a lookup
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    table = DispatchTable.from_schemas(
        case.contract, torch.device("cpu"), [(s, r, s.workloads) for s, r in served], min_speedup=1.0
    )
    return dataclasses.replace(table, fast_key=FastKeyCodec.from_contract(case.contract).to_fast_key)


def test_a_bucketed_axis_matches_its_power_of_two_range() -> None:
    """fused_moe's n_tokens: 3000 matches 3072 (both 12 bits), not 4096 or 2047."""
    case = OpCase.of("fused_moe@1.0")
    codec = FastKeyCodec.from_contract(case.contract)
    assert _key(codec, case, 3000) == _key(codec, case, 3072)
    assert _key(codec, case, 3000) != _key(codec, case, 4096)
    assert _key(codec, case, 3000) != _key(codec, case, 2047)


def test_an_axis_is_any_size_or_bucketed_not_both() -> None:
    with pytest.raises(ValueError, match="not both"):
        AxisSpec(kind="int", description="x", lo=1, hi=8, is_var=True, any_size=True, bucketed=True)
    with pytest.raises(ValueError, match="var axis"):
        AxisSpec(kind="int", description="x", lo=1, hi=8, bucketed=True)


@pytest.mark.parametrize(
    ("slow_n_tokens", "served"),
    [(None, True), (3500, False), (5000, True)],
    ids=["no_slow_workload", "slow_workload_in_the_range", "slow_workload_in_another_range"],
)
def test_a_schema_serves_a_range_only_if_each_of_its_workloads_there_qualifies(
    monkeypatch, slow_n_tokens, served
) -> None:
    """A schema measured at 2100 and 3072 tokens (2.0x) serves 3000, between them in their range [2048, 4096); a
    third workload of its in that range at 0.5x leaves 3000 unserved, one in another range does not."""
    case = OpCase.of("fused_moe@1.0")
    runnable = _Runnable(monkeypatch, True)
    measured = {2100: 2.0, 3072: 2.0} | ({slow_n_tokens: 0.5} if slow_n_tokens else {})
    table = _table(monkeypatch, case, (_schema(case, measured), runnable))
    assert table.lookup(_point(case, 3000))[2] is (runnable if served else FallbackReason.NOT_QUALIFIED)


@pytest.mark.parametrize("range_safe", [True, False], ids=["range_safe", "not_range_safe"])
def test_a_measured_value_is_served_by_its_own_workload_first(monkeypatch, range_safe) -> None:
    """Measured at 3072 (2.0x) and 3500 (0.5x): its range is unserved, yet 3072 itself is served, 3500 is not, and
    3000 (measured by none) falls back, with the range's reason for a RANGE_SAFE kernel; a kernel that does not
    declare it serves its measured values alone."""
    case = OpCase.of("fused_moe@1.0")
    runnable = _Runnable(monkeypatch, range_safe)
    table = _table(monkeypatch, case, (_schema(case, {3072: 2.0, 3500: 0.5}), runnable))
    assert table.lookup(_point(case, 3072))[2] is runnable
    assert table.lookup(_point(case, 3500))[2] is FallbackReason.NOT_QUALIFIED
    assert table.lookup(_point(case, 3000))[2] is (
        FallbackReason.NOT_QUALIFIED if range_safe else FallbackReason.NO_WORKLOAD
    )


def test_a_kernel_not_range_safe_serves_its_measured_values_alone(monkeypatch) -> None:
    """Measured at 2100 and 3072 (2.0x) without RANGE_SAFE: 2100 and 3072 are served, 3000 between them is not."""
    case = OpCase.of("fused_moe@1.0")
    runnable = _Runnable(monkeypatch, False)
    table = _table(monkeypatch, case, (_schema(case, {2100: 2.0, 3072: 2.0}), runnable))
    assert table.lookup(_point(case, 2100))[2] is runnable
    assert table.lookup(_point(case, 3072))[2] is runnable
    assert table.lookup(_point(case, 3000))[2] is FallbackReason.NO_WORKLOAD


def test_a_range_is_served_only_between_its_lowest_and_highest_measured_values(monkeypatch) -> None:
    """Measured at 2100 and 3072 (both 2.0x, the range [2048, 4096)): 3000 is served, 2050 and 3500 (outside what
    was measured, where a kernel's last tile class may end) fall back."""
    case = OpCase.of("fused_moe@1.0")
    runnable = _Runnable(monkeypatch, True)
    table = _table(monkeypatch, case, (_schema(case, {2100: 2.0, 3072: 2.0}), runnable))
    assert table.lookup(_point(case, 3000))[2] is runnable
    assert table.lookup(_point(case, 2100))[2] is runnable
    assert table.lookup(_point(case, 3500))[2] is FallbackReason.NO_WORKLOAD
    assert table.lookup(_point(case, 2050))[2] is FallbackReason.NO_WORKLOAD
