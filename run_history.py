"""Session-local immutable metadata for optimization attempts.

Run history intentionally stores metadata only. Complete solution rows remain owned by
``SolutionStore`` and are referenced through immutable ``run_solution_*`` handles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from problem_definition import BONSAI_PROBLEM_ID


class RunHistoryError(Exception):
    """Base class for predictable run-history failures."""


class RunNotFoundError(RunHistoryError):
    """Raised when a run identifier is not registered."""


class RunAlreadyFinalizedError(RunHistoryError):
    """Raised when a reserved run identifier is finalized more than once."""


@dataclass(frozen=True)
class OptimizationRunRecord:
    """Immutable metadata for one completed or failed optimization attempt."""

    run_id: str
    problem_id: str
    run_kind: str
    status: str
    fast: bool
    parameter_overrides: tuple[tuple[str, object], ...]
    scenario_handle: str | None = None
    solution_handle: str | None = None
    current_alias: str | None = None
    error_code: str | None = None

    @property
    def overrides(self) -> Dict[str, object]:
        return dict(self.parameter_overrides)


@dataclass(frozen=True)
class _PendingRun:
    run_id: str
    problem_id: str
    run_kind: str
    fast: bool
    parameter_overrides: tuple[tuple[str, object], ...]
    scenario_handle: str | None = None


class RunHistoryStore:
    """In-memory sequential run metadata for one agent/application session."""

    def __init__(self, *, default_problem_id: str = BONSAI_PROBLEM_ID) -> None:
        if not isinstance(default_problem_id, str) or not default_problem_id.strip():
            raise ValueError("RunHistoryStore default_problem_id must be non-empty text.")
        self._default_problem_id = default_problem_id.strip()
        self._next_sequence = 1
        self._pending: Dict[str, _PendingRun] = {}
        self._records: Dict[str, OptimizationRunRecord] = {}

    @staticmethod
    def _normalize_overrides(overrides: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
        return tuple(sorted(dict(overrides).items()))

    @staticmethod
    def solution_handle_for(run_id: str) -> str:
        if not isinstance(run_id, str) or not run_id.startswith("run_"):
            raise RunHistoryError(f"Invalid run id: {run_id!r}")
        sequence = run_id.removeprefix("run_")
        if len(sequence) != 6 or not sequence.isdigit():
            raise RunHistoryError(f"Invalid run id: {run_id!r}")
        return f"run_solution_{sequence}"

    def begin(
        self,
        *,
        run_kind: str,
        fast: bool,
        parameter_overrides: Mapping[str, object],
        scenario_handle: str | None = None,
        problem_id: str | None = None,
    ) -> str:
        if run_kind not in {"base", "scenario"}:
            raise RunHistoryError(f"Unsupported run kind: {run_kind!r}")
        if run_kind == "scenario" and not scenario_handle:
            raise RunHistoryError("Scenario runs require a scenario handle.")
        if run_kind == "base" and scenario_handle is not None:
            raise RunHistoryError("Base runs cannot have a scenario handle.")
        resolved_problem_id = self._default_problem_id if problem_id is None else problem_id
        if not isinstance(resolved_problem_id, str) or not resolved_problem_id.strip():
            raise RunHistoryError("Runs require a non-empty problem_id.")

        run_id = f"run_{self._next_sequence:06d}"
        self._next_sequence += 1
        self._pending[run_id] = _PendingRun(
            run_id=run_id,
            problem_id=resolved_problem_id.strip(),
            run_kind=run_kind,
            fast=bool(fast),
            parameter_overrides=self._normalize_overrides(parameter_overrides),
            scenario_handle=scenario_handle,
        )
        return run_id

    def _finalize(
        self,
        run_id: str,
        *,
        status: str,
        solution_handle: str | None,
        current_alias: str | None,
        error_code: str | None,
    ) -> OptimizationRunRecord:
        if run_id in self._records:
            raise RunAlreadyFinalizedError(f"Run already finalized: {run_id}")
        try:
            pending = self._pending.pop(run_id)
        except KeyError as exc:
            raise RunNotFoundError(f"Run is not pending: {run_id}") from exc

        if status == "completed":
            if not solution_handle or error_code is not None:
                raise RunHistoryError("Completed runs require a solution handle and no error code.")
        elif status == "failed":
            if solution_handle is not None:
                raise RunHistoryError("Failed runs cannot reference a historical solution handle.")
        else:
            raise RunHistoryError(f"Unsupported run status: {status!r}")

        record = OptimizationRunRecord(
            run_id=pending.run_id,
            problem_id=pending.problem_id,
            run_kind=pending.run_kind,
            status=status,
            fast=pending.fast,
            parameter_overrides=pending.parameter_overrides,
            scenario_handle=pending.scenario_handle,
            solution_handle=solution_handle,
            current_alias=current_alias,
            error_code=error_code,
        )
        self._records[run_id] = record
        return record

    def complete(
        self,
        run_id: str,
        *,
        solution_handle: str,
        current_alias: str,
    ) -> OptimizationRunRecord:
        return self._finalize(
            run_id,
            status="completed",
            solution_handle=solution_handle,
            current_alias=current_alias,
            error_code=None,
        )

    def fail(
        self,
        run_id: str,
        *,
        error_code: str,
        current_alias: str | None = None,
    ) -> OptimizationRunRecord:
        return self._finalize(
            run_id,
            status="failed",
            solution_handle=None,
            current_alias=current_alias,
            error_code=error_code,
        )

    def get(self, run_id: str) -> OptimizationRunRecord:
        try:
            return self._records[run_id]
        except KeyError as exc:
            raise RunNotFoundError(f"Run is not registered: {run_id}") from exc

    def list(self) -> tuple[OptimizationRunRecord, ...]:
        return tuple(self._records.values())

    def __contains__(self, run_id: object) -> bool:
        return isinstance(run_id, str) and run_id in self._records


__all__ = [
    "OptimizationRunRecord",
    "RunAlreadyFinalizedError",
    "RunHistoryError",
    "RunHistoryStore",
    "RunNotFoundError",
]
