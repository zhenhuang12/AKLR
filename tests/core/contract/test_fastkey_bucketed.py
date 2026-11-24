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

# one class over fused_moe's whole n_tokens axis, [1, 65536]: one build serves every count
_WHOLE_AXIS = {"n_tokens": (65536,)}


class _Runnable:
    """A kernel whose module declares RANGE_SAFE and, unless None, RANGE_CLASSES as given."""

    def __init__(self, monkeypatch, range_safe, classes=None):
        name = f"tests_range_{id(self)}"
        module = types.ModuleType(name)
        module.RANGE_SAFE = range_safe
        if classes is not None:
            module.RANGE_CLASSES = classes
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
    runnable = _Runnable(monkeypatch, True, _WHOLE_AXIS)
    measured = {2100: 2.0, 3072: 2.0} | ({slow_n_tokens: 0.5} if slow_n_tokens else {})
    table = _table(monkeypatch, case, (_schema(case, measured), runnable))
    assert table.lookup(_point(case, 3000))[2] is (runnable if served else FallbackReason.NOT_QUALIFIED)


@pytest.mark.parametrize("range_safe", [True, False], ids=["range_safe", "not_range_safe"])
def test_a_measured_value_is_served_by_its_own_workload_first(monkeypatch, range_safe) -> None:
    """Measured at 3072 (2.0x) and 3500 (0.5x): its range is unserved, yet 3072 itself is served, 3500 is not, and
    3000 (measured by none) falls back, with the range's reason for a RANGE_SAFE kernel; a kernel that does not
    declare it serves its measured values alone."""
    case = OpCase.of("fused_moe@1.0")
    runnable = _Runnable(monkeypatch, range_safe, _WHOLE_AXIS)
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
    runnable = _Runnable(monkeypatch, True, _WHOLE_AXIS)
    table = _table(monkeypatch, case, (_schema(case, {2100: 2.0, 3072: 2.0}), runnable))
    assert table.lookup(_point(case, 3000))[2] is runnable
    assert table.lookup(_point(case, 2100))[2] is runnable
    assert table.lookup(_point(case, 3500))[2] is FallbackReason.NO_WORKLOAD
    assert table.lookup(_point(case, 2050))[2] is FallbackReason.NO_WORKLOAD


@pytest.mark.parametrize(
    ("speedup", "reason"),
    [(None, FallbackReason.INACCURATE), (0.5, FallbackReason.NOT_QUALIFIED)],
    ids=["inaccurate", "slow"],
)
def test_a_value_a_workload_measured_is_decided_by_it_not_by_a_range(monkeypatch, speedup, reason) -> None:
    """One schema serves [2048, 4096) between 2100 and 3072 (2.0x); another measured 2500 and failed there: 2500
    falls back with that workload's reason, not served by the range, while 3000, which no workload holds, is."""
    case = OpCase.of("fused_moe@1.0")
    wide, other = _Runnable(monkeypatch, True, _WHOLE_AXIS), _Runnable(monkeypatch, True, _WHOLE_AXIS)
    table = _table(
        monkeypatch,
        case,
        (_schema(case, {2100: 2.0, 3072: 2.0}), wide),
        (_schema(case, {2500: speedup}), other),
    )
    assert table.lookup(_point(case, 2500))[2] is reason
    assert table.lookup(_point(case, 3000))[2] is wide


def _narrow_and_wide(monkeypatch, case):
    """The table of a 3.0x kernel measured at 2300 and 2600 and a 2.0x one at 2100 and 3072, both RANGE_SAFE in
    [2048, 4096): (the table, the narrow kernel, the wide one)."""
    narrow, wide = _Runnable(monkeypatch, True, _WHOLE_AXIS), _Runnable(monkeypatch, True, _WHOLE_AXIS)
    table = _table(
        monkeypatch,
        case,
        (_schema(case, {2300: 3.0, 2600: 3.0}), narrow),
        (_schema(case, {2100: 2.0, 3072: 2.0}), wide),
    )
    return table, narrow, wide


@pytest.mark.parametrize(
    ("n_tokens", "serving"),
    [(2400, "narrow"), (3000, "wide"), (2050, None)],
    ids=["both_measured_around_it", "only_the_wide_one_around_it", "none_around_it"],
)
def test_a_range_is_served_by_the_best_kernel_measured_around_the_value(
    monkeypatch, n_tokens, serving
) -> None:
    """A faster kernel measured over a narrower part of a range hides no slower one measured around a value: 2400
    takes the 3.0x kernel (2300-2600), 3000 the 2.0x one (2100-3072), 2050 neither."""
    case = OpCase.of("fused_moe@1.0")
    table, narrow, wide = _narrow_and_wide(monkeypatch, case)
    expected = {"narrow": narrow, "wide": wide, None: FallbackReason.NO_WORKLOAD}[serving]
    assert table.lookup(_point(case, n_tokens))[2] is expected


def test_a_range_kernel_is_warm_only_once_it_ran(monkeypatch) -> None:
    """Inside a graph capture a range's kernel serves only after an eager call of its own: once 2400 ran (the 3.0x
    kernel), 2500 is served, 3000 (the 2.0x kernel, never run) is not warm."""
    case = OpCase.of("fused_moe@1.0")
    table, narrow, _ = _narrow_and_wide(monkeypatch, case)
    _, match, runnable = table.lookup(_point(case, 2400))
    assert runnable is narrow
    table.warmed_keys.add(match)  # as KernelDispatcher.dispatch does after an eager call
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert table.lookup(_point(case, 2500))[2] is narrow
    assert table.lookup(_point(case, 3000))[2] is FallbackReason.NOT_WARMED


@pytest.mark.parametrize(
    ("contract_id", "plus_one_axis"),
    [("sparse_prefill_mxfp8@1.0", "n_offsets"), ("topk_transform_paged@1.0", "n_plan_rows")],
    ids=["sparse_prefill_mxfp8_n_offsets", "topk_transform_paged_n_plan_rows"],
)
def test_an_axis_bound_to_n_tokens_plus_one_does_not_split_its_range(contract_id, plus_one_axis) -> None:
    """An axis the contract holds to n_tokens + 1 (CSR offsets, plan rows) leaves 3000 and 3072 tokens one range key,
    so a RANGE_SAFE kernel measured at 3072 may serve 3000."""
    case = OpCase.of(contract_id)
    codec = FastKeyCodec.from_contract(case.contract)

    def key(n_tokens):
        axes = {**case.point.axes, "n_tokens": n_tokens, plus_one_axis: n_tokens + 1}
        return codec.match_key(codec.to_fast_key(case.point.model_copy(update={"axes": axes})))

    assert key(3000) == key(3072)


def test_a_kernel_declaring_range_safe_alone_serves_its_measured_values_alone(monkeypatch) -> None:
    """RANGE_SAFE without RANGE_CLASSES (every RANGE_SAFE kernel committed before the class declaration): measured at
    2100 and 3072 (2.0x), it serves those two, and 3000 between them falls back."""
    case = OpCase.of("fused_moe@1.0")
    runnable = _Runnable(monkeypatch, True)
    table = _table(monkeypatch, case, (_schema(case, {2100: 2.0, 3072: 2.0}), runnable))
    assert table.lookup(_point(case, 2100))[2] is runnable
    assert table.lookup(_point(case, 3072))[2] is runnable
    assert table.lookup(_point(case, 3000))[2] is FallbackReason.NO_WORKLOAD


@pytest.mark.parametrize(
    ("n_tokens", "served"),
    [(2200, True), (2450, False), (2600, False), (3000, False), (3500, True), (3700, False)],
    ids=[
        "between_the_first_classs_measured_ends",
        "past_the_first_classs_last_measured",
        "in_the_second_class_below_its_first_measured",
        "in_the_second_class_measured_once",
        "between_the_third_classs_measured_ends",
        "above_the_last_class",
    ],
)
def test_a_range_is_served_class_by_class(monkeypatch, n_tokens, served) -> None:
    """Classes (.., 2500], (2500, 3100], (3100, 3600] of [2048, 4096), measured at 2100 and 2400 (first), 3072
    (second), 3200 and 3550 (third), all 2.0x: an unmeasured count is served only between the measured ends of its own
    class, never in a class measured once, nor above the last class."""
    case = OpCase.of("fused_moe@1.0")
    runnable = _Runnable(monkeypatch, True, {"n_tokens": (2500, 3100, 3600)})
    measured = {2100: 2.0, 2400: 2.0, 3072: 2.0, 3200: 2.0, 3550: 2.0}
    table = _table(monkeypatch, case, (_schema(case, measured), runnable))
    assert table.lookup(_point(case, n_tokens))[2] is (runnable if served else FallbackReason.NO_WORKLOAD)


def test_a_slow_workload_closes_its_class_alone(monkeypatch) -> None:
    """Classes (.., 2500] and (2500, 65536]: a 0.5x workload at 2300 leaves the first class unserved (2200 falls
    back), while the second, measured at 2600 and 3072 (2.0x), serves 3000."""
    case = OpCase.of("fused_moe@1.0")
    runnable = _Runnable(monkeypatch, True, {"n_tokens": (2500, 65536)})
    measured = {2100: 2.0, 2300: 0.5, 2400: 2.0, 2600: 2.0, 3072: 2.0}
    table = _table(monkeypatch, case, (_schema(case, measured), runnable))
    assert table.lookup(_point(case, 2200))[2] is FallbackReason.NO_WORKLOAD
    assert table.lookup(_point(case, 3000))[2] is runnable


def test_a_class_is_warm_only_once_a_call_of_it_ran(monkeypatch) -> None:
    """Inside a graph capture a class serves only after an eager call in it: once 2200 (first class) ran, 2300 is
    served, 3000 (second class, its build never run) is not warm."""
    case = OpCase.of("fused_moe@1.0")
    runnable = _Runnable(monkeypatch, True, {"n_tokens": (2500, 65536)})
    table = _table(monkeypatch, case, (_schema(case, {2100: 2.0, 2400: 2.0, 2600: 2.0, 3072: 2.0}), runnable))
    _, match, served = table.lookup(_point(case, 2200))
    assert served is runnable
    table.warmed_keys.add(match)  # as KernelDispatcher.dispatch does after an eager call
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert table.lookup(_point(case, 2300))[2] is runnable
    assert table.lookup(_point(case, 3000))[2] is FallbackReason.NOT_WARMED


@pytest.mark.parametrize(
    "classes",
    [
        {"n_tokens": (3000, 2500)},
        {"n_tokens": ()},
        {"n_tokens": [65536]},
        {"model_dim": (65536,)},
        {"n_tokens": (65536,), "model_dim": (16384,)},
        "n_tokens",
    ],
    ids=["descending", "empty", "a_list", "another_axis", "an_axis_more", "not_a_mapping"],
)
def test_a_malformed_class_declaration_serves_measured_values_alone(monkeypatch, classes) -> None:
    """A declaration dispatch cannot read as each bucketed axis's ascending class bounds (audit's SA6 rejects it) is
    no declaration: 2100 and 3072 are served, 3000 is not."""
    case = OpCase.of("fused_moe@1.0")
    runnable = _Runnable(monkeypatch, True, classes)
    table = _table(monkeypatch, case, (_schema(case, {2100: 2.0, 3072: 2.0}), runnable))
    assert table.lookup(_point(case, 3072))[2] is runnable
    assert table.lookup(_point(case, 3000))[2] is FallbackReason.NO_WORKLOAD
