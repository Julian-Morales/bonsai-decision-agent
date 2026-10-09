"""Concrete application adapter for the existing deterministic Bonsai problem."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Mapping, Protocol, Sequence

from optimization_problem import OptimizationProblem, OptimizationProblemError, default_bonsai_packaging_problem
from problem_adapter import OptimizationExecutionOutcome, business_contexts_compatible
from problem_definition import BONSAI_PROBLEM_ID, OptimizationProblemSpec, bonsai_problem_spec


class BonsaiExecutionContext(Protocol):
    catalog_path: Path
    specs_path: Path
    operations_path: Path
    procurement_path: Path
    output_dir: Path


def _bonsai_tools_module():
    # Lazy import keeps the generic adapter contract and non-Bonsai agents free from the
    # Bonsai optimizer dependency until a concrete Bonsai operation is requested.
    import bonsai_tools
    return bonsai_tools


def _read_solution_rows(path: Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


class BonsaiProblemAdapter:
    """Delegate audit/optimization/validation/comparison to existing Bonsai services."""

    def __init__(self, context: BonsaiExecutionContext) -> None:
        self._catalog_path = Path(context.catalog_path)
        self._specs_path = Path(context.specs_path)
        self._operations_path = Path(context.operations_path)
        self._procurement_path = Path(context.procurement_path)
        self._output_dir = Path(context.output_dir)

    @property
    def problem_id(self) -> str:
        return BONSAI_PROBLEM_ID

    @property
    def problem(self) -> OptimizationProblem:
        return default_bonsai_packaging_problem()

    @property
    def spec(self) -> OptimizationProblemSpec:
        return bonsai_problem_spec(self.problem)

    @property
    def has_golden_reference(self) -> bool:
        return True

    @property
    def supports_fast_mode(self) -> bool:
        return True

    def resolve_problem(self, overrides: Mapping[str, object]) -> OptimizationProblem:
        return self.problem.with_overrides(dict(overrides))

    def audit_data(self) -> Dict[str, object]:
        return _bonsai_tools_module().audit_bonsai_data(
            self._catalog_path,
            self._specs_path,
            self._operations_path,
            self._procurement_path,
        )

    def get_baseline(self) -> Dict[str, object]:
        return _bonsai_tools_module().get_baseline_costs(
            self._catalog_path,
            self._specs_path,
            self._operations_path,
            self._procurement_path,
        )

    def run_optimizer(
        self,
        *,
        fast: bool,
        parameter_overrides: Mapping[str, object],
    ) -> Dict[str, object]:
        return _bonsai_tools_module().run_optimizer(
            self._catalog_path,
            self._specs_path,
            self._operations_path,
            self._procurement_path,
            self._output_dir,
            fast=bool(fast),
            quiet=True,
            parameter_overrides=dict(parameter_overrides),
        )

    @staticmethod
    def _optimizer_failure_code(result: Mapping[str, object]) -> str:
        validation = result.get("final_validation") if isinstance(result.get("final_validation"), Mapping) else {}
        final_solution = result.get("final_solution") if isinstance(result.get("final_solution"), Mapping) else {}
        if not bool(result.get("success", False)):
            return "optimizer_failed"
        if not bool(validation.get("valid", False)) or not bool(final_solution.get("independently_validated", False)):
            return "independent_validation_failed"
        return "solution_registration_failed"

    def _compact_optimizer_result(self, result: Mapping[str, object]) -> Dict[str, object]:
        baseline = result.get("baseline") if isinstance(result.get("baseline"), Mapping) else {}
        validation = result.get("final_validation") if isinstance(result.get("final_validation"), Mapping) else {}
        greedy = result.get("greedy") if isinstance(result.get("greedy"), Mapping) else {}
        milp = result.get("milp") if isinstance(result.get("milp"), Mapping) else {}
        final_solution = result.get("final_solution") if isinstance(result.get("final_solution"), Mapping) else {}
        history = result.get("local_search_history")
        history_count = len(history) if isinstance(history, list) else 0
        problem = result.get("problem") if isinstance(result.get("problem"), Mapping) else {}
        problem_values = problem.get("values") if isinstance(problem.get("values"), Mapping) else {}
        problem_parameters = problem.get("parameters") if isinstance(problem.get("parameters"), Mapping) else {}
        raw_business = problem.get("business_parameters") if isinstance(problem.get("business_parameters"), Mapping) else {}
        if raw_business:
            effective_business = self._validated_business_context(raw_business)
        else:
            business_overrides = {
                name: problem_values[name]
                for name, definition in self.problem.parameters.items()
                if definition.category == "business" and name in problem_values
            }
            effective_business = self.resolve_problem(business_overrides).business_parameters
        reference_business = self.problem.business_parameters
        reference_compatible = business_contexts_compatible(effective_business, reference_business)
        return {
            "success": bool(result.get("success", False)),
            "problem": {
                "name": problem.get("name"),
                "values": dict(problem_values),
                "parameter_categories": {
                    name: metadata.get("category")
                    for name, metadata in problem_parameters.items()
                    if isinstance(metadata, Mapping)
                },
                "capabilities": problem.get("capabilities"),
            },
            "reference_context_compatible": reference_compatible,
            "effective_business_context": dict(effective_business),
            "reference_business_context": dict(reference_business),
            "baseline": {"total": baseline.get("total")} if reference_compatible else None,
            "optimizer_candidate": {"independently_validated": False},
            "validated_final": {
                "valid": bool(validation.get("valid", False)),
                "error_count": validation.get("error_count"),
                "packaging": validation.get("packaging"),
                "freight": validation.get("freight"),
                "total": validation.get("total"),
                "saving": validation.get("saving") if reference_compatible else None,
                "saving_pct": validation.get("saving_pct") if reference_compatible else None,
                "n_types": validation.get("n_types"),
                "thicknesses": validation.get("thicknesses"),
                "independently_validated": bool(final_solution.get("independently_validated", False)),
            },
            "stages": {
                "greedy": {"total": greedy.get("total"), "n_types": greedy.get("n_types")},
                "milp": {"status": milp.get("status"), "total": milp.get("total"), "types": milp.get("types")},
                "local_search_improvement_count": history_count,
            },
        }

    def interpret_optimizer_result(
        self,
        result: Mapping[str, object],
    ) -> OptimizationExecutionOutcome:
        validation = result.get("final_validation") if isinstance(result.get("final_validation"), Mapping) else {}
        final_solution = result.get("final_solution") if isinstance(result.get("final_solution"), Mapping) else {}
        validated_success = (
            bool(result.get("success", False))
            and bool(validation.get("valid", False))
            and bool(final_solution.get("independently_validated", False))
        )
        rows: tuple[Mapping[str, object], ...] = ()
        if validated_success:
            artifact = final_solution.get("artifact")
            if isinstance(artifact, str) and artifact:
                try:
                    loaded = _read_solution_rows(Path(artifact))
                    rows = tuple(loaded)
                except (OSError, TypeError):
                    rows = ()
            validated_success = bool(rows)
        return OptimizationExecutionOutcome(
            validated_success=validated_success,
            rows=rows,
            compact_result=self._compact_optimizer_result(result),
            failure_code=None if validated_success else self._optimizer_failure_code(result),
        )

    def _validated_business_context(
        self,
        business_context: Mapping[str, object],
    ) -> Dict[str, object]:
        business_names = {
            name
            for name, definition in self.problem.parameters.items()
            if definition.category == "business"
        }
        if set(business_context) != business_names:
            raise OptimizationProblemError("Stored business context does not match this problem.")
        resolved = self.problem.with_overrides(dict(business_context))
        return resolved.business_parameters

    def _pallet_cost(self, business_context: Mapping[str, object]) -> float:
        validated = self._validated_business_context(business_context)
        definition = self.problem.parameters.get("pallet_cost")
        if definition is None or definition.category != "business":
            raise RuntimeError(
                "Bonsai problem definition does not declare business parameter 'pallet_cost'."
            )
        return float(definition.validate(validated[definition.name]))

    def validate_solution(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        business_context: Mapping[str, object],
    ) -> Dict[str, object]:
        return _bonsai_tools_module().validate_solution(
            rows,
            self._catalog_path,
            self._specs_path,
            self._operations_path,
            pallet_cost=self._pallet_cost(business_context),
        )

    def compact_validation_result(
        self,
        handle: str,
        report: Mapping[str, object],
    ) -> Dict[str, object]:
        errors = report.get("errors") if isinstance(report.get("errors"), list) else []
        return {
            "success": True,
            "handle": handle,
            "valid": bool(report.get("valid", False)),
            "n_types": report.get("n_types"),
            "thicknesses": report.get("thicknesses"),
            "packaging": report.get("packaging"),
            "freight": report.get("freight"),
            "total": report.get("total"),
            "saving": report.get("saving"),
            "saving_pct": report.get("saving_pct"),
            "error_count": report.get("error_count"),
            "error_sample": errors[:5],
        }

    def compare_solutions(
        self,
        solution_a: Sequence[Mapping[str, object]],
        solution_b: Sequence[Mapping[str, object]],
        *,
        business_context: Mapping[str, object],
    ) -> Dict[str, object]:
        return _bonsai_tools_module().compare_solutions(
            solution_a,
            solution_b,
            self._catalog_path,
            self._specs_path,
            self._operations_path,
            pallet_cost=self._pallet_cost(business_context),
        )

    @staticmethod
    def _compact_comparison_solution(summary: Mapping[str, object]) -> Dict[str, object]:
        errors = summary.get("errors") if isinstance(summary.get("errors"), list) else []
        return {
            "valid": bool(summary.get("valid", False)),
            "n_types": summary.get("n_types"),
            "thicknesses": summary.get("thicknesses"),
            "packaging": summary.get("packaging"),
            "freight": summary.get("freight"),
            "total": summary.get("total"),
            "saving_pct": summary.get("saving_pct"),
            "error_count": summary.get("error_count"),
            "error_sample": errors[:5],
        }

    def compact_comparison_result(
        self,
        solution_a_handle: str,
        solution_b_handle: str,
        comparison: Mapping[str, object],
    ) -> Dict[str, object]:
        summary_a = comparison.get("solution_a") if isinstance(comparison.get("solution_a"), Mapping) else {}
        summary_b = comparison.get("solution_b") if isinstance(comparison.get("solution_b"), Mapping) else {}
        return {
            "success": True,
            "solution_a_handle": solution_a_handle,
            "solution_b_handle": solution_b_handle,
            "comparable": bool(comparison.get("comparable", False)),
            "solution_a": self._compact_comparison_solution(summary_a),
            "solution_b": self._compact_comparison_solution(summary_b),
            "delta_b_minus_a": comparison.get("delta_b_minus_a"),
        }


__all__ = ["BonsaiExecutionContext", "BonsaiProblemAdapter"]
