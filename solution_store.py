"""Session-local registry for complete deterministic solution rows and provenance.

The store is intentionally application-only: it knows solution handles, row payloads,
problem provenance, and immutable business/evaluation context metadata, but no domain
validation rules, optimizer behavior, providers, or files.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Dict, Mapping, Sequence, Tuple

from problem_definition import BONSAI_PROBLEM_ID


GOLDEN_REFERENCE = "golden_reference"
LATEST_OPTIMIZER = "latest_optimizer"
SUPPORTED_SOLUTION_HANDLES = (GOLDEN_REFERENCE, LATEST_OPTIMIZER)
_SCENARIO_SOLUTION_HANDLE = re.compile(r"^scenario_solution_[a-z0-9_]+$")
_RUN_SOLUTION_HANDLE = re.compile(r"^run_solution_\d{6}$")


def is_supported_solution_handle(handle: object) -> bool:
    return isinstance(handle, str) and (
        handle in SUPPORTED_SOLUTION_HANDLES
        or _SCENARIO_SOLUTION_HANDLE.fullmatch(handle) is not None
        or _RUN_SOLUTION_HANDLE.fullmatch(handle) is not None
    )


class SolutionStoreError(Exception):
    """Base class for predictable solution-registry failures."""


class UnsupportedSolutionHandleError(SolutionStoreError):
    """Raised when a caller uses a handle outside the solution-registry contract."""


class SolutionNotFoundError(SolutionStoreError):
    """Raised when a supported handle is not currently registered."""


class SolutionHandleExistsError(SolutionStoreError):
    """Raised when a handle cannot be overwritten under its lifecycle rules."""


@dataclass(frozen=True)
class StoredSolution:
    """Private immutable container tying rows to their evaluation provenance.

    Nested payloads are defensively copied on both registration and retrieval. The
    class itself is not exposed to the LLM.
    """

    problem_id: str
    business_context: Mapping[str, object] | None
    rows: Tuple[Mapping[str, object], ...]


class SolutionStore:
    """In-memory owner of complete solution rows for one application session.

    ``business_context`` is metadata needed to reproduce deterministic solution
    economics. ``problem_id`` identifies which deterministic problem adapter is allowed
    to validate/compare those rows. Solver-only configuration intentionally does not
    live here.
    """

    def __init__(self) -> None:
        self._solutions_by_handle: Dict[str, StoredSolution] = {}

    @staticmethod
    def _require_supported_handle(handle: str) -> None:
        if not is_supported_solution_handle(handle):
            raise UnsupportedSolutionHandleError(f"Unsupported solution handle: {handle!r}")

    @staticmethod
    def _copy_rows(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
        if isinstance(rows, (str, bytes)):
            raise TypeError("Solution rows must be a sequence of mappings, not text.")
        normalized: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("Each solution row must be a mapping.")
            normalized.append(dict(row))
        return tuple(deepcopy(normalized))

    @staticmethod
    def _copy_business_context(
        business_context: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        if business_context is None:
            return None
        if not isinstance(business_context, Mapping):
            raise TypeError("Solution business context must be a mapping or None.")
        return deepcopy(dict(business_context))

    @staticmethod
    def _normalize_problem_id(problem_id: str) -> str:
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise TypeError("Solution problem_id must be non-empty text.")
        return problem_id.strip()

    @classmethod
    def _stored_solution(
        cls,
        rows: Sequence[Mapping[str, object]],
        *,
        problem_id: str,
        business_context: Mapping[str, object] | None,
    ) -> StoredSolution:
        return StoredSolution(
            problem_id=cls._normalize_problem_id(problem_id),
            business_context=cls._copy_business_context(business_context),
            rows=cls._copy_rows(rows),
        )

    def register(
        self,
        handle: str,
        rows: Sequence[Mapping[str, object]],
        *,
        replace: bool = False,
        business_context: Mapping[str, object] | None = None,
        problem_id: str = BONSAI_PROBLEM_ID,
    ) -> None:
        """Register rows and provenance under a supported handle.

        ``problem_id`` defaults to the accepted Bonsai ID only for backward-compatible
        direct callers. Adapter-driven application paths always pass it explicitly.
        """
        self._require_supported_handle(handle)

        if handle in self._solutions_by_handle:
            if (
                handle == GOLDEN_REFERENCE
                or _RUN_SOLUTION_HANDLE.fullmatch(handle) is not None
                or not replace
            ):
                raise SolutionHandleExistsError(f"Solution handle already registered: {handle}")

        stored = self._stored_solution(
            rows,
            problem_id=problem_id,
            business_context=business_context,
        )
        self._solutions_by_handle[handle] = stored

    def register_run_result(
        self,
        historical_handle: str,
        current_alias: str,
        rows: Sequence[Mapping[str, object]],
        *,
        business_context: Mapping[str, object],
        problem_id: str = BONSAI_PROBLEM_ID,
    ) -> None:
        """Atomically register one immutable run snapshot and update its current alias."""
        if _RUN_SOLUTION_HANDLE.fullmatch(historical_handle) is None:
            raise UnsupportedSolutionHandleError(
                f"Historical solution handle is not a generated run handle: {historical_handle!r}"
            )
        self._require_supported_handle(current_alias)
        if current_alias == GOLDEN_REFERENCE or _RUN_SOLUTION_HANDLE.fullmatch(current_alias) is not None:
            raise UnsupportedSolutionHandleError(
                f"Unsupported current alias for a run result: {current_alias!r}"
            )
        if historical_handle in self._solutions_by_handle:
            raise SolutionHandleExistsError(
                f"Historical solution handle already registered: {historical_handle}"
            )

        stored = self._stored_solution(
            rows,
            problem_id=problem_id,
            business_context=business_context,
        )
        # Independent copies preserve alias/history provenance even if a future alias moves.
        self._solutions_by_handle[historical_handle] = deepcopy(stored)
        self._solutions_by_handle[current_alias] = deepcopy(stored)

    def _get_stored(self, handle: str) -> StoredSolution:
        self._require_supported_handle(handle)
        try:
            return self._solutions_by_handle[handle]
        except KeyError as exc:
            raise SolutionNotFoundError(f"Solution handle is not registered: {handle}") from exc

    def get(self, handle: str) -> list[dict[str, object]]:
        """Return an independent copy of the rows registered under ``handle``."""
        stored = self._get_stored(handle)
        return deepcopy([dict(row) for row in stored.rows])

    def get_business_context(self, handle: str) -> dict[str, object] | None:
        """Return a defensive copy of deterministic evaluation context."""
        stored = self._get_stored(handle)
        return self._copy_business_context(stored.business_context)

    def get_problem_id(self, handle: str) -> str:
        """Return the immutable optimization-problem provenance for ``handle``."""
        return self._get_stored(handle).problem_id

    def __contains__(self, handle: object) -> bool:
        return isinstance(handle, str) and handle in self._solutions_by_handle


__all__ = [
    "GOLDEN_REFERENCE",
    "LATEST_OPTIMIZER",
    "SUPPORTED_SOLUTION_HANDLES",
    "StoredSolution",
    "is_supported_solution_handle",
    "SolutionHandleExistsError",
    "SolutionNotFoundError",
    "SolutionStore",
    "SolutionStoreError",
    "UnsupportedSolutionHandleError",
]
