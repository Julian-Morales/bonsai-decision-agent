"""Production Capacity Planning adapter for the generic decision-agent application layer."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

from optimization_problem import OptimizationProblem, OptimizationProblemError
from problem_adapter import OptimizationExecutionOutcome, business_contexts_compatible
from problem_definition import OptimizationProblemSpec
from production_data import (
    DEFAULT_PRODUCTION_DATA_DIR,
    ProductionPlanningData,
    audit_production_data,
    load_production_data,
)
from production_optimizer import optimize_production_plan
from production_problem import (
    PRODUCTION_CAPACITY_PLANNING_PROBLEM_ID,
    default_production_capacity_planning_problem,
    production_capacity_planning_spec,
)
from production_validator import validate_production_solution


class ProductionPlanningAdapter:
    """Concrete deterministic adapter for the bundled production-planning domain."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = Path(data_dir or DEFAULT_PRODUCTION_DATA_DIR)
        self._data: ProductionPlanningData | None = None

    @property
    def problem_id(self) -> str:
        return PRODUCTION_CAPACITY_PLANNING_PROBLEM_ID

    @property
    def problem(self) -> OptimizationProblem:
        return default_production_capacity_planning_problem()

    @property
    def spec(self) -> OptimizationProblemSpec:
        return production_capacity_planning_spec(self.problem)

    @property
    def has_golden_reference(self) -> bool:
        return False

    @property
    def supports_fast_mode(self) -> bool:
        # Production has one deterministic MILP pipeline; the shared fast flag has no
        # alternate execution behavior for this adapter.
        return False

    @property
    def data(self) -> ProductionPlanningData:
        if self._data is None:
            self._data = load_production_data(self._data_dir)
        return self._data

    def resolve_problem(self, overrides: Mapping[str, object]) -> OptimizationProblem:
        return self.problem.with_overrides(dict(overrides))

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
        return self.problem.with_overrides(dict(business_context)).business_parameters

    def audit_data(self) -> Dict[str, object]:
        audit = audit_production_data(self.data)
        baseline = self.get_baseline()
        return {
            **audit,
            "baseline_valid": baseline.get("valid"),
            "problem_id": self.problem_id,
        }

    def get_baseline(self) -> Dict[str, object]:
        business = self.problem.business_parameters
        report = validate_production_solution(
            self.data.current_plan,
            self.data,
            demand_multiplier=float(business["demand_multiplier"]),
            overtime_cost_multiplier=float(business["overtime_cost_multiplier"]),
        )
        return {
            "valid": report["valid"],
            "production_cost": report["production_cost"],
            "overtime_cost": report["overtime_cost"],
            "total_cost": report["total_cost"],
            "total_units": report["total_units"],
            "overtime_hours": report["overtime_hours"],
            "source": "production_demo/current_plan.csv",
        }

    def run_optimizer(
        self,
        *,
        fast: bool,
        parameter_overrides: Mapping[str, object],
    ) -> Dict[str, object]:
        # ``fast`` is part of the shared tool contract. This deliberately small domain
        # has no second local-search stage, so it does not change the mathematical model.
        resolved = self.resolve_problem(parameter_overrides)
        business = resolved.business_parameters
        solver = resolved.solver_parameters
        optimized = optimize_production_plan(
            self.data,
            demand_multiplier=float(business["demand_multiplier"]),
            overtime_cost_multiplier=float(business["overtime_cost_multiplier"]),
            milp_time=float(solver["milp_time"]),
            mip_gap=float(solver["mip_gap"]),
        )
        rows = optimized.get("solution_rows") if isinstance(optimized.get("solution_rows"), list) else []
        validation = validate_production_solution(
            rows,
            self.data,
            demand_multiplier=float(business["demand_multiplier"]),
            overtime_cost_multiplier=float(business["overtime_cost_multiplier"]),
        )
        validated = bool(optimized.get("success", False) and validation.get("valid", False))
        return {
            "success": bool(optimized.get("success", False)),
            "independently_validated": validated,
            "problem_id": self.problem_id,
            "problem_values": dict(resolved.values),
            "business_context": dict(business),
            "solver_parameters": dict(solver),
            "fast": bool(fast),
            "solver": {
                "status": optimized.get("solver_status"),
                "message": optimized.get("solver_message"),
                "objective": optimized.get("objective"),
                "variable_count": optimized.get("variable_count"),
                "constraint_count": optimized.get("constraint_count"),
            },
            "solution_rows": rows,
            "validation": validation,
        }

    def interpret_optimizer_result(
        self,
        result: Mapping[str, object],
    ) -> OptimizationExecutionOutcome:
        validation = result.get("validation") if isinstance(result.get("validation"), Mapping) else {}
        rows_value = result.get("solution_rows")
        rows = tuple(dict(row) for row in rows_value) if isinstance(rows_value, list) else ()
        validated_success = bool(
            result.get("success", False)
            and result.get("independently_validated", False)
            and validation.get("valid", False)
            and rows
        )
        problem_values = result.get("problem_values") if isinstance(result.get("problem_values"), Mapping) else {}
        effective_business = (
            dict(result.get("business_context"))
            if isinstance(result.get("business_context"), Mapping)
            else self.resolve_problem({
                name: problem_values[name]
                for name, definition in self.problem.parameters.items()
                if definition.category == "business" and name in problem_values
            }).business_parameters
        )
        reference_business = self.problem.business_parameters
        reference_compatible = business_contexts_compatible(effective_business, reference_business)
        compact = {
            "success": bool(result.get("success", False)),
            "problem": {
                "name": self.problem.name,
                "values": dict(problem_values),
                "parameter_categories": {
                    parameter.name: parameter.category for parameter in self.spec.parameters
                },
                "capabilities": dict(self.problem.capabilities),
            },
            "reference_context_compatible": reference_compatible,
            "effective_business_context": dict(effective_business),
            "reference_business_context": dict(reference_business),
            # Do not expose default-context baseline economics beside an incompatible
            # result: that invites a non-like-for-like delta to be synthesized by the LLM.
            "baseline": self.get_baseline() if reference_compatible else None,
            "validated_final": {
                "valid": bool(validation.get("valid", False)),
                "error_count": validation.get("error_count"),
                "production_cost": validation.get("production_cost"),
                "overtime_cost": validation.get("overtime_cost"),
                "total_cost": validation.get("total_cost"),
                "total_units": validation.get("total_units"),
                "overtime_hours": validation.get("overtime_hours"),
                "independently_validated": bool(result.get("independently_validated", False)),
            },
            "solver": dict(result.get("solver")) if isinstance(result.get("solver"), Mapping) else {},
        }
        if not bool(result.get("success", False)):
            failure = "optimizer_failed"
        elif not bool(result.get("independently_validated", False)):
            failure = "independent_validation_failed"
        else:
            failure = "solution_registration_failed"
        return OptimizationExecutionOutcome(
            validated_success=validated_success,
            rows=rows if validated_success else (),
            compact_result=compact,
            failure_code=None if validated_success else failure,
        )

    def validate_solution(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        business_context: Mapping[str, object],
    ) -> Dict[str, object]:
        business = self._validated_business_context(business_context)
        return validate_production_solution(
            rows,
            self.data,
            demand_multiplier=float(business["demand_multiplier"]),
            overtime_cost_multiplier=float(business["overtime_cost_multiplier"]),
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
            "production_cost": report.get("production_cost"),
            "overtime_cost": report.get("overtime_cost"),
            "total_cost": report.get("total_cost"),
            "total_units": report.get("total_units"),
            "overtime_hours": report.get("overtime_hours"),
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
        report_a = self.validate_solution(solution_a, business_context=business_context)
        report_b = self.validate_solution(solution_b, business_context=business_context)
        comparable = bool(report_a.get("valid", False) and report_b.get("valid", False))
        delta = None
        if comparable:
            delta = {
                "production_cost": round(float(report_b["production_cost"]) - float(report_a["production_cost"]), 8),
                "overtime_cost": round(float(report_b["overtime_cost"]) - float(report_a["overtime_cost"]), 8),
                "total_cost": round(float(report_b["total_cost"]) - float(report_a["total_cost"]), 8),
                "total_units": round(float(report_b["total_units"]) - float(report_a["total_units"]), 8),
                "overtime_hours": round(float(report_b["overtime_hours"]) - float(report_a["overtime_hours"]), 8),
            }
        return {
            "comparable": comparable,
            "solution_a": report_a,
            "solution_b": report_b,
            "delta_b_minus_a": delta,
        }

    @staticmethod
    def _compact_comparison_solution(report: Mapping[str, object]) -> Dict[str, object]:
        errors = report.get("errors") if isinstance(report.get("errors"), list) else []
        return {
            "valid": bool(report.get("valid", False)),
            "production_cost": report.get("production_cost"),
            "overtime_cost": report.get("overtime_cost"),
            "total_cost": report.get("total_cost"),
            "total_units": report.get("total_units"),
            "overtime_hours": report.get("overtime_hours"),
            "error_count": report.get("error_count"),
            "error_sample": errors[:5],
        }

    def compact_comparison_result(
        self,
        solution_a_handle: str,
        solution_b_handle: str,
        comparison: Mapping[str, object],
    ) -> Dict[str, object]:
        a = comparison.get("solution_a") if isinstance(comparison.get("solution_a"), Mapping) else {}
        b = comparison.get("solution_b") if isinstance(comparison.get("solution_b"), Mapping) else {}
        return {
            "success": True,
            "solution_a_handle": solution_a_handle,
            "solution_b_handle": solution_b_handle,
            "comparable": bool(comparison.get("comparable", False)),
            "solution_a": self._compact_comparison_solution(a),
            "solution_b": self._compact_comparison_solution(b),
            "delta_b_minus_a": comparison.get("delta_b_minus_a"),
        }


__all__ = ["ProductionPlanningAdapter"]
