"""Business-level deterministic interface for Bonsai Corp.

This module is the stable boundary intended for future application/LLM adapters.
It composes the existing deterministic data, optimization, and validation layers without
exposing optimizer implementation details.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

import bonsai_data
from run_optimization import OptimizationConfig, optimize_portfolio
from optimization_problem import OptimizationProblemError, default_bonsai_packaging_problem
from validate_submission import validate_solution_rows


def _json_safe(value):
    """Recursively convert supported deterministic outputs to JSON-native values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    raise TypeError(f"Valor no serializable por la capa de tools: {type(value).__name__}")


def audit_bonsai_data(
    catalog_path: Path,
    specs_path: Path,
    operations_path: Path,
    procurement_path: Path,
) -> Dict[str, object]:
    """Return the deterministic source-data and Procurement consistency audit."""
    prepared = bonsai_data.load_inputs(
        Path(catalog_path),
        Path(specs_path),
        Path(operations_path),
        Path(procurement_path),
    )
    summary, detail = bonsai_data.audit_procurement(prepared)

    mismatches = detail.loc[
        detail["observed_price"].notna() & ~detail["matches_operations_tier"]
    ]
    result = {
        **summary,
        "procurement_mismatch_count": int(len(mismatches)),
        "procurement_mismatches": mismatches.to_dict(orient="records"),
    }
    return _json_safe(result)


def get_baseline_costs(
    catalog_path: Path,
    specs_path: Path,
    operations_path: Path,
    procurement_path: Path,
) -> Dict[str, object]:
    """Return the baseline packaging and freight economics used by the optimizer."""
    prepared = bonsai_data.load_inputs(
        Path(catalog_path),
        Path(specs_path),
        Path(operations_path),
        Path(procurement_path),
    )
    baseline = bonsai_data.baseline_from_operations(prepared.operations)
    return _json_safe({**baseline, "source": "operaciones_planta"})


def run_optimizer(
    catalog_path: Path,
    specs_path: Path,
    operations_path: Path,
    procurement_path: Path,
    output_dir: Path,
    *,
    milp_time: float = 30.0,
    mip_gap: float = 5e-4,
    extra_candidates: int = 40,
    lns_time: float = 0.2,
    lns_rounds: int = 2,
    target_count: int = 40,
    target_time: float = 0.05,
    quiet: bool = False,
    fast: bool = False,
    parameter_overrides: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Run the existing optimizer with validated problem/solver parameter overrides."""
    problem = default_bonsai_packaging_problem()
    try:
        problem = problem.with_overrides(parameter_overrides or {})
    except OptimizationProblemError as exc:
        return {
            "success": False,
            "error": {"code": "invalid_parameter_override", "message": str(exc)},
            "problem": {
                "name": problem.name,
                "capabilities": problem.capabilities,
            },
        }

    solver = problem.solver_parameters
    # Explicit legacy keyword arguments remain the service-level source for callers that
    # already use them; an override for a solver parameter takes precedence.
    config = OptimizationConfig(
        milp_time=float(solver["milp_time"] if parameter_overrides and "milp_time" in parameter_overrides else milp_time),
        mip_gap=float(solver["mip_gap"] if parameter_overrides and "mip_gap" in parameter_overrides else mip_gap),
        extra_candidates=int(solver["extra_candidates"] if parameter_overrides and "extra_candidates" in parameter_overrides else extra_candidates),
        lns_time=float(solver["lns_time"] if parameter_overrides and "lns_time" in parameter_overrides else lns_time),
        lns_rounds=int(solver["lns_rounds"] if parameter_overrides and "lns_rounds" in parameter_overrides else lns_rounds),
        target_count=int(solver["target_count"] if parameter_overrides and "target_count" in parameter_overrides else target_count),
        target_time=float(solver["target_time"] if parameter_overrides and "target_time" in parameter_overrides else target_time),
        quiet=quiet,
        fast=fast,
    )
    result = optimize_portfolio(
        catalog_path=Path(catalog_path),
        specs_path=Path(specs_path),
        operations_path=Path(operations_path),
        procurement_path=Path(procurement_path),
        output_dir=Path(output_dir),
        config=config,
        problem=problem,
    )

    result_problem = getattr(result, "problem", None) or default_bonsai_packaging_problem()
    response = {
        "success": result.success,
        "problem": {
            "name": result_problem.name,
            "parameters": {
                name: {
                    "category": definition.category,
                    "default": definition.default,
                    "configurable": definition.configurable,
                    "description": definition.description,
                }
                for name, definition in result_problem.parameters.items()
            },
            "values": dict(result_problem.values),
            "business_parameters": result_problem.business_parameters,
            "solver_parameters": result_problem.solver_parameters,
            "capabilities": result_problem.capabilities,
            "data_requirements": result_problem.data_requirements,
        },
        "config": result.config,
        "baseline": result.baseline,
        "data_audit": result.data_audit,
        "greedy": result.greedy,
        "milp": result.milp,
        "local_search_history": result.local_search_history,
        "optimizer_candidate": {
            "artifact": result.candidate_solution_path,
            "independently_validated": False,
        },
        "final_validation": result.final_validation,
        "final_summary": result.final_summary,
        "final_solution": {
            "artifact": result.final_solution_path,
            "independently_validated": bool(result.final_validation.get("valid", False)),
        },
        "artifacts": result.artifacts,
    }
    return _json_safe(response)


def validate_solution(
    rows: Sequence[Mapping[str, object]],
    catalog_path: Path,
    specs_path: Path,
    operations_path: Path,
    *,
    mode: str = "faq",
    pallet_cost: float | None = None,
) -> Dict[str, object]:
    """Independently validate proposed rows under an explicit business context.

    When no pallet cost is supplied by a caller, resolve the canonical default from
    ``OptimizationProblem`` rather than duplicating a numeric default in this adapter.
    """
    if pallet_cost is None:
        problem = default_bonsai_packaging_problem()
        pallet_cost = float(problem.business_parameters[problem.parameters["pallet_cost"].name])
    report = validate_solution_rows(
        rows,
        Path(catalog_path),
        Path(specs_path),
        Path(operations_path),
        mode=mode,
        pallet_cost=pallet_cost,
    )
    return _json_safe(report)


def _comparison_solution_summary(report: Mapping[str, object]) -> Dict[str, object]:
    valid = bool(report["valid"])
    return {
        "valid": valid,
        "n_types": report.get("n_types"),
        "thicknesses": report.get("thicknesses"),
        # Economics from an infeasible solution are intentionally not exposed as
        # comparable business metrics.
        "packaging": report.get("packaging") if valid else None,
        "freight": report.get("freight") if valid else None,
        "total": report.get("total") if valid else None,
        "saving_pct": report.get("saving_pct") if valid else None,
        "error_count": report.get("error_count"),
        "errors": report.get("errors", []),
    }


def compare_solutions(
    solution_a: Sequence[Mapping[str, object]],
    solution_b: Sequence[Mapping[str, object]],
    catalog_path: Path,
    specs_path: Path,
    operations_path: Path,
    *,
    mode: str = "faq",
    pallet_cost: float | None = None,
) -> Dict[str, object]:
    """Validate and compare two solutions under one explicit common business context."""
    if pallet_cost is None:
        problem = default_bonsai_packaging_problem()
        pallet_cost = float(problem.business_parameters[problem.parameters["pallet_cost"].name])
    report_a = validate_solution_rows(
        solution_a,
        Path(catalog_path),
        Path(specs_path),
        Path(operations_path),
        mode=mode,
        pallet_cost=pallet_cost,
    )
    report_b = validate_solution_rows(
        solution_b,
        Path(catalog_path),
        Path(specs_path),
        Path(operations_path),
        mode=mode,
        pallet_cost=pallet_cost,
    )

    comparable = bool(report_a["valid"] and report_b["valid"])
    result: Dict[str, object] = {
        "comparable": comparable,
        "solution_a": _comparison_solution_summary(report_a),
        "solution_b": _comparison_solution_summary(report_b),
        "delta_b_minus_a": None,
    }

    if comparable:
        result["delta_b_minus_a"] = {
            "n_types": int(report_b["n_types"]) - int(report_a["n_types"]),
            "packaging": float(report_b["packaging"]) - float(report_a["packaging"]),
            "freight": float(report_b["freight"]) - float(report_a["freight"]),
            "total": float(report_b["total"]) - float(report_a["total"]),
            "saving_pct_points": float(report_b["saving_pct"]) - float(report_a["saving_pct"]),
        }

    return _json_safe(result)


__all__ = [
    "audit_bonsai_data",
    "get_baseline_costs",
    "run_optimizer",
    "validate_solution",
    "compare_solutions",
]
