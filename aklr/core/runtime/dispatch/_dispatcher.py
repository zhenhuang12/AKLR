# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Created by OpenAI

from __future__ import annotations

import ast
import bisect
import functools
import inspect
import logging
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import torch

from aklr.core.contract.closure import (
    compile_closure,
    fill_expr,
    fill_statements,
)
from aklr.core.contract.fastkey import (
    AllocOutputsFnGenerator,
    CallArgs,
    FastKey,
    FastKeyCodec,
    FastKeyFnGenerator,
)
from aklr.core.contract.op import OpContract
from aklr.core.contract.spec import TOOLCHAINS, WorkloadSpec, axes_text
from aklr.core.platform.env import ENV_DISABLE
from aklr.core.platform.toolchain import toolchain_version
from aklr.core.runtime.builder import Runnable, RunnableMetadata
from aklr.core.runtime.dispatch._trace import trace

if TYPE_CHECKING:
    from aklr.core.runtime.dispatch._recorder import CallRecorder

__all__ = ["DispatchTable", "FallbackReason", "KernelDispatcher", "kernel_failure_hint"]

_LOGGER = logging.getLogger(__name__)


def kernel_failure_hint(contract_id: str, language: str) -> str:
    """What a failing kernel's message adds: its toolchain here, and the switch that
    turns its op off."""
    toolchain = TOOLCHAINS[language]
    version = toolchain_version(toolchain) or "not installed"
    return f"{toolchain} {version} here; {ENV_DISABLE}={contract_id} turns this op off"


class FallbackReason(str, Enum):
    """Why an api call did not run a AKLR kernel; the caller then runs its own."""

    NO_SCHEMA = "no_schema"
    # AKLR_DISABLE
    DISABLED = "disabled"
    # fast_key returned None: no Point describes the call
    OUT_OF_CONTRACT = "out_of_contract"
    # keyed, but no loaded schema has a workload of this Point
    NO_WORKLOAD = "no_workload"
    # keyed, but no candidate has a speedup
    NOT_MEASURED = "not_measured"
    INACCURATE = "inaccurate"
    # measured passed, but none reaches AKLR_MIN_SPEEDUP
    NOT_QUALIFIED = "not_qualified"
    # only inside a graph capture: its Point never ran outside one
    NOT_WARMED = "not_warmed"


_FALLBACK_PRECEDENCE: Final = (
    FallbackReason.NOT_QUALIFIED,
    FallbackReason.INACCURATE,
    FallbackReason.NOT_MEASURED,
)


def _range_classes(runnable: Runnable, axes: tuple[str, ...]) -> tuple[tuple[int, ...], ...] | None:
    """Per bucketed axis (``axes``), the upper bounds of the kernel's classes on it, as its module declares them:
    ``RANGE_SAFE = True`` (it builds one variant per class, none per value) and ``RANGE_CLASSES``, each bucketed
    axis's ascending inclusive upper bounds (a class runs from above the bound before it, the first from the axis's
    lowest value). None without both, or where they do not read so (audit's SA6 rejects such a declaration): the
    kernel serves its measured values alone."""
    metadata = getattr(runnable, "metadata", None)
    name = None if metadata is None else metadata.misc.get("module_name")
    module = sys.modules.get(name) if name else None
    declared = getattr(module, "RANGE_CLASSES", None)
    if getattr(module, "RANGE_SAFE", False) is not True or not isinstance(declared, Mapping):
        return None
    if set(declared) != set(axes):
        return None
    bounds = tuple(declared[a] for a in axes)
    for b in bounds:
        if not (isinstance(b, tuple) and b and all(type(x) is int for x in b)):
            return None
        if any(x >= y for x, y in zip(b, b[1:], strict=False)):
            return None
    return bounds


def _class_of(
    classes: tuple[tuple[int, ...], ...], key: FastKey, indices: tuple[int, ...]
) -> tuple[int, ...] | None:
    """Per bucketed axis, the index of the class ``key``'s value falls in; None above an axis's last class."""
    found = tuple(bisect.bisect_left(bounds, key[i]) for bounds, i in zip(classes, indices, strict=True))
    return None if any(c == len(bounds) for c, bounds in zip(found, classes, strict=True)) else found


def _note_reasons(
    reasons: dict[FastKey, FallbackReason], key: FastKey, failing: Iterable[WorkloadSpec]
) -> None:
    """Each failing workload's reason at ``key``, the first in _FALLBACK_PRECEDENCE kept."""
    for workload in failing:
        if workload.accuracy == "failed":
            reason = FallbackReason.INACCURATE
        elif workload.speedup is None:
            reason = FallbackReason.NOT_MEASURED
        else:
            reason = FallbackReason.NOT_QUALIFIED
        reasons[key] = min(reasons.get(key, reason), reason, key=_FALLBACK_PRECEDENCE.index)


# per bucketed axis, the lowest and highest value a schema measured in a range's class
_Span = tuple[tuple[int, int], ...]


class _RangeCandidate(NamedTuple):
    """A kernel serving one of its classes within a power-of-two range of a bucketed axis."""

    # an unmeasured value is served only inside its span, the values its schema measured in that class of the range
    # (a kernel's tile classes end at its last measured value, past which it may build per value or not run at all)
    span: _Span
    runnable: Runnable
    # what a call of it marks warm: (the range key, its place in the range), one per class of a kernel: an eager call
    # in the class builds what every value of it runs
    warm_key: FastKey


@dataclass(frozen=True, slots=True, eq=False)
class DispatchTable:
    """One contract's committed kernels on one device, keyed by each Point's FastKey as
    ``match_key`` (``FastKeyCodec.match_key``) leaves it; with a bucketed axis, by each
    workload's own value (``exact_key``) and, for kernels declaring their classes, by range (``ranges``)."""

    # the closures FastKeyFnGenerator and AllocOutputsFnGenerator make
    fast_key: Callable[[CallArgs], FastKey | None]
    alloc_outputs: Callable[[FastKey, CallArgs], Any]
    runnables: dict[FastKey, Runnable]
    warmed_keys: set[FastKey]
    # keys with candidates but no qualified one; a plain dict, as runnables
    fallback_reasons: dict[FastKey, FallbackReason]
    match_key: Callable[[FastKey], FastKey]
    # a contract with a bucketed axis: a call's exact key (``FastKeyCodec.exact_key``), which alone decides a call
    # some workload was measured at (served where one qualifies, else its reason); None without one
    exact_key: Callable[[FastKey], FastKey] | None = None
    # a range key (``match_key`` with a bucketed axis) -> each class of each kernel declaring its classes
    # (``_range_classes``) whose schema qualifies at all its workloads in that class of the range, best ranked first:
    # a value no workload holds takes the first whose span holds it
    ranges: dict[FastKey, tuple[_RangeCandidate, ...]] = field(default_factory=dict)
    bucketed_indices: tuple[int, ...] = ()

    @classmethod
    def from_schemas(
        cls,
        contract: OpContract,
        device: torch.device,
        schema_runnables: Iterable[tuple[OpContract, Runnable, Iterable[WorkloadSpec]]],
        *,
        min_speedup: float,
    ) -> DispatchTable:
        """The table serving, per Point of the given workloads."""
        codec = FastKeyCodec.from_contract(contract)
        best: dict[FastKey, tuple[tuple[bool, float, str], Runnable]] = {}
        # a range key -> (rank, span, runnable) of each kernel's class serving it
        in_range: dict[FastKey, list[tuple[tuple[bool, float, str], _Span, Runnable]]] = {}
        fallback_reasons: dict[FastKey, FallbackReason] = {}
        indices = codec.bucketed_indices
        bucketed = bool(indices)
        axes = tuple(tuple(contract.domain.axes)[i] for i in indices)
        for schema, runnable, workloads in schema_runnables:
            # workloads differing in any_size axes alone share a key: the schema serves it
            # only if each of them qualifies, at the lowest rank among them; with a bucketed
            # axis, each workload's own value is a key, served before its range's, which only a
            # kernel declaring its classes serves (``_range_classes``), class by class
            classes = _range_classes(runnable, axes) if bucketed else None
            keyed: dict[FastKey, list[WorkloadSpec]] = {}
            # (a range key, a class of the kernel in it) -> its workloads, with their keys
            cells: dict[tuple[FastKey, tuple[int, ...]], list[tuple[FastKey, WorkloadSpec]]] = {}
            for workload in workloads:
                fast_key = codec.to_fast_key(schema.point(workload.axes))
                if not bucketed:
                    keyed.setdefault(codec.match_key(fast_key), []).append(workload)
                    continue
                keyed.setdefault(codec.exact_key(fast_key), []).append(workload)
                in_class = None if classes is None else _class_of(classes, fast_key, indices)
                if in_class is not None:
                    cells.setdefault((codec.match_key(fast_key), in_class), []).append((fast_key, workload))
            for key, same in keyed.items():
                failing = [w for w in same if not w.qualifies(min_speedup)]
                if not failing:
                    rank = min(map(schema.rank_key, same))
                    # strict: an equal rank shares a content hash, i.e.
                    if key not in best or rank > best[key][0]:
                        best[key] = (rank, runnable)
                _note_reasons(fallback_reasons, key, failing)
            # a range keeps every class of every kernel serving it, each with its span, where each workload of the
            # schema in that class of the range qualifies: one ranked higher over a narrower span hides no other
            # where only the other was measured around; a failing class's reasons are its range's
            for (key, _), members in cells.items():
                failing = [w for _, w in members if not w.qualifies(min_speedup)]
                if not failing:
                    span = tuple(
                        (min(k[i] for k, _ in members), max(k[i] for k, _ in members)) for i in indices
                    )
                    rank = min(schema.rank_key(w) for _, w in members)
                    in_range.setdefault(key, []).append((rank, span, runnable))
                _note_reasons(fallback_reasons, key, failing)
        served_ranges = {
            key: tuple(
                _RangeCandidate(span, runnable, (key, i))
                # stable: of an equal rank (one content hash), the first
                for i, (_, span, runnable) in enumerate(sorted(found, key=lambda c: c[0], reverse=True))
            )
            for key, found in in_range.items()
        }
        return cls(
            FastKeyFnGenerator(codec, device).generate(),
            AllocOutputsFnGenerator(codec, device).generate(),
            {key: runnable for key, (_, runnable) in best.items()},
            set(),
            {
                key: reason
                for key, reason in fallback_reasons.items()
                if key not in best and key not in served_ranges
            },
            codec.match_key,
            codec.exact_key if bucketed else None,
            served_ranges,
            codec.bucketed_indices,
        )

    @property
    def closed_reason(self) -> FallbackReason:
        """Why a dispatcher given this table closes instead (no runnable, no recorder)."""
        return min(
            self.fallback_reasons.values(),
            key=_FALLBACK_PRECEDENCE.index,
            default=FallbackReason.NOT_MEASURED,
        )

    def lookup(self, call: CallArgs) -> tuple[FastKey | None, FastKey | None, Runnable | FallbackReason]:
        """``call``'s key, its match and the kernel serving it, else why none does."""
        key = self.fast_key(call)
        if key is None:
            return None, None, FallbackReason.OUT_OF_CONTRACT
        if self.exact_key is None:
            # no bucketed axis: one key, as before ranges
            match = self.match_key(key)
            runnable = self.runnables.get(match)
        else:
            # a value a workload measured is decided by it alone: its range serves only values no workload holds
            match = self.exact_key(key)
            runnable = self.runnables.get(match)
            if runnable is None and match not in self.fallback_reasons:
                match = self.match_key(key)
                # a range: its best kernel whose schema measured values on both sides of the call's
                for candidate in self.ranges.get(match, ()):
                    if all(
                        lo <= key[i] <= hi
                        for i, (lo, hi) in zip(self.bucketed_indices, candidate.span, strict=True)
                    ):
                        match, runnable = candidate.warm_key, candidate.runnable
                        break
        if runnable is None:
            reason = self.fallback_reasons.get(match) or (
                FallbackReason.NO_WORKLOAD
                if self.runnables or self.fallback_reasons
                else FallbackReason.NO_SCHEMA
            )
            return key, match, reason
        if match not in self.warmed_keys and torch.cuda.is_current_stream_capturing():
            return key, match, FallbackReason.NOT_WARMED
        return key, match, runnable


class KernelDispatcher:
    """One contract's dispatch: the api function (``make_api_fn``) calls ``dispatch``."""

    __slots__ = (
        "closed_reason",
        "contract",
        "has_warned_not_warmed",
        "recorder",
        "table",
    )
    contract: OpContract
    table: DispatchTable | None  # None: every call returns closed_reason, unkeyed
    closed_reason: FallbackReason  # read only while table is None
    recorder: CallRecorder | None  # export on; None whenever table is None
    has_warned_not_warmed: bool

    def __init__(self, contract: OpContract) -> None:
        self.contract = contract
        self.closed_reason = FallbackReason.NO_SCHEMA
        self.has_warned_not_warmed = False
        self.recorder = None
        self.table = None

    def dispatch(self, call: CallArgs) -> Any:
        """The api's result, or a ``FallbackReason``.

        The result is the outputs: one alone, else a tuple in ``contract.outputs`` order,
        the caller's buffers where ``call`` holds them, else allocated first. The kernel
        fills them.
        """
        table = self.table
        if table is None:
            if (tracer := trace()) is not None and tracer.is_op:
                tracer.op(self.contract, None, self.closed_reason)
            return self.closed_reason
        key, match, runnable = table.lookup(call)
        if isinstance(runnable, FallbackReason):
            result = runnable
            if result is FallbackReason.NOT_WARMED and not self.has_warned_not_warmed:
                self.has_warned_not_warmed = True  # one warning per dispatcher
                _LOGGER.warning(
                    "%s: %s was first called inside a graph capture: it falls "
                    "back there and in every replay; call each Point once "
                    "outside a capture first (later Points are not named)",
                    self.contract.contract_id,
                    axes_text(FastKeyCodec.from_contract(self.contract).to_point(key).axes),
                )
        else:  # a warmed hit, or the first eager call (it may compile)
            result = table.alloc_outputs(key, call)
            try:
                runnable(**call)
            except Exception:
                point = FastKeyCodec.from_contract(self.contract).to_point(key)
                _LOGGER.error(
                    "%s: the kernel serving %s at %s failed (%s)",
                    self.contract.contract_id,
                    self.contract.name_at(point),
                    axes_text(point.axes),
                    kernel_failure_hint(self.contract.contract_id, runnable.metadata.build_type),
                )
                raise
            table.warmed_keys.add(match)
        if (tracer := trace()) is not None and tracer.is_op:
            tracer.op(self.contract, key, runnable)
        if self.recorder is not None:
            self.recorder.count(
                self.contract,
                key,
                call,
                result if isinstance(result, FallbackReason) else None,
            )
        return result

    def fallback_reason(self, call: CallArgs) -> FallbackReason | None:
        """``None`` if ``dispatch`` would serve ``call``, else its reason, counted as
        ``dispatch`` counts it; nothing runs."""
        served = self.serving(call)
        return served if isinstance(served, FallbackReason) else None

    def serving(self, call: CallArgs) -> RunnableMetadata | FallbackReason:
        """The kernel ``dispatch`` would serve ``call`` with, else its reason, counted as
        ``dispatch`` counts it; nothing runs."""
        table = self.table
        if table is None:
            if (tracer := trace()) is not None and tracer.is_op:
                tracer.op(self.contract, None, self.closed_reason)
            return self.closed_reason
        key, _, runnable = table.lookup(call)
        if not isinstance(runnable, FallbackReason):
            return runnable.metadata
        if (tracer := trace()) is not None and tracer.is_op:
            tracer.op(self.contract, key, runnable)
        if self.recorder is not None:
            self.recorder.count(self.contract, key, call, runnable)
        return runnable

    def set_table(self, table: DispatchTable, recorder: CallRecorder | None = None) -> None:
        """Serve by ``table``, counting into ``recorder`` unless ``None``."""
        self.recorder = recorder
        self.table = table

    def close(self, reason: FallbackReason) -> None:
        """Return ``reason`` for every call, uncounted even while exporting."""
        self.closed_reason = reason
        self.recorder = None
        self.table = None

    def make_api_fn(self, fn: Callable) -> Callable:
        """``fn`` checked against ``contract`` (``check_api_signature``), and the api function
        replacing it, whose ``fallback_reason`` and ``serving`` ask without serving."""
        self.contract.check_api_signature(fn)
        api_fn = self._generate(fn, "dispatch")
        api_fn.fallback_reason = self._generate(fn, "fallback_reason")
        api_fn.serving = self._generate(fn, "serving")
        return api_fn

    def _generate(self, fn: Callable, method: str) -> Callable:
        """A function of ``fn``'s signature calling ``method`` on the call it forms."""
        code = fn.__code__
        names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        args, kwonly = names[: code.co_argcount], names[code.co_argcount :]
        params = inspect.signature(fn).parameters
        contract_names = {p.name for p in self.contract.api_params}
        # Defaults go through scope, not as constants: an enum is no constant.
        defaults = {p.name: p.default for p in self.contract.api_params}
        defaults |= {n: params[n].default for n in names if n not in contract_names}
        module = ast.parse(f"def f():\n    return _dispatcher.{method}()")
        node = module.body[0]
        node.name = fn.__name__
        node.args.args = [ast.arg(name) for name in args]
        node.args.kwonlyargs = [ast.arg(name) for name in kwonly]
        node.args.kw_defaults = [None] * len(kwonly)
        guards = [
            fill_statements(
                "if P is not D and (D is None or P != D):\n    return _out_of_contract",
                P=name,
                D=fill_expr("_d[N]", N=ast.Constant(name)),
            )[0]
            for name in names
            if name not in contract_names
        ]
        call = ast.Dict(
            [ast.Constant(p.name) for p in self.contract.api_params],
            [
                (
                    ast.Name(p.name, ast.Load())
                    if p.name in names
                    else fill_expr("_d[N]", N=ast.Constant(p.name))
                )
                for p in self.contract.api_params
            ],
        )
        node.body[0].value.args = [call]
        node.body[:0] = guards
        scope = {
            "__builtins__": {},
            "_dispatcher": self,
            "_d": defaults,
            "_out_of_contract": FallbackReason.OUT_OF_CONTRACT,
        }
        origin = f"<api:{self.contract.contract_id}>"
        generated = compile_closure(module, fn.__name__, scope, origin)
        functools.update_wrapper(generated, fn)
        generated.__defaults__ = fn.__defaults__
        generated.__kwdefaults__ = fn.__kwdefaults__
        return generated
