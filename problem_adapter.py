"""Application-level optimization-problem adapter contract.

The contract intentionally owns no optimization mathematics. Concrete domain adapters
translate their deterministic domain services into a small common application envelope
while keeping domain-specific summaries inside the adapter boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, Sequence, Tuple

from optimization_problem import OptimizationProblem
from problem_definition import OptimizationProblemSpec


@dataclass(frozen=True)
class OptimizationExecutionOutcome:
    """Common application envelope for one deterministic optimizer result.

    Complete rows remain application data and are never part of ``compact_result``.
    ``validated_success`` means the domain's independent validator accepted the result.
    """

    validated_success: bool
    rows: Tuple[Mapping[str, object], ...]
    compact_result: Mapping[str, object]
    failure_code: str | None = None


def business_contexts_compatible(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> bool:
    """Return whether two stored business/evaluation contexts are like-for-like.

    This intentionally uses exact normalized mapping equality. The application already
    validates values through ``OptimizationProblem`` before storing them, so comparison
    policy must not invent tolerances or parameter-specific exceptions here.
    """
    return dict(left) == dict(right)


class OptimizationProblemAdapter(Protocol):
    """Minimal application contract for one deterministic optimization domain."""

    @property
    def problem_id(self) -> str: ...

    @property
    def problem(self) -> OptimizationProblem: ...

    @property
    def spec(self) -> OptimizationProblemSpec: ...

    @property
    def has_golden_reference(self) -> bool: ...

    @property
    def supports_fast_mode(self) -> bool: ...

    def resolve_problem(self, overrides: Mapping[str, object]) -> OptimizationProblem: ...

    def audit_data(self) -> Dict[str, object]: ...

    def get_baseline(self) -> Dict[str, object]: ...

    def run_optimizer(
        self,
        *,
        fast: bool,
        parameter_overrides: Mapping[str, object],
    ) -> Dict[str, object]: ...

    def interpret_optimizer_result(
        self,
        result: Mapping[str, object],
    ) -> OptimizationExecutionOutcome: ...

    def validate_solution(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        business_context: Mapping[str, object],
    ) -> Dict[str, object]: ...

    def compact_validation_result(
        self,
        handle: str,
        report: Mapping[str, object],
    ) -> Dict[str, object]: ...

    def compare_solutions(
        self,
        solution_a: Sequence[Mapping[str, object]],
        solution_b: Sequence[Mapping[str, object]],
        *,
        business_context: Mapping[str, object],
    ) -> Dict[str, object]: ...

    def compact_comparison_result(
        self,
        solution_a_handle: str,
        solution_b_handle: str,
        comparison: Mapping[str, object],
    ) -> Dict[str, object]: ...


__all__ = ["OptimizationExecutionOutcome", "OptimizationProblemAdapter", "business_contexts_compatible"]
