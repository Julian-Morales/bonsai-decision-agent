"""Small deterministic optimizer for the Production Capacity Planning demo domain."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from production_data import ProductionPlanningData


def optimize_production_plan(
    data: ProductionPlanningData,
    *,
    demand_multiplier: float,
    overtime_cost_multiplier: float,
    milp_time: float,
    mip_gap: float,
) -> dict[str, object]:
    """Minimize variable production plus overtime premium under capacity constraints.

    Variables are regular and overtime production units for each compatible
    product-line pair. Variables are continuous because this capacity-planning demo is
    an aggregate planning model, not a job-level scheduling model.
    """
    edges = tuple(sorted(data.compatibility))
    n_edges = len(edges)
    n_vars = n_edges * 2
    reg_index = {edge: i for i, edge in enumerate(edges)}
    ot_index = {edge: i + n_edges for i, edge in enumerate(edges)}

    c = np.zeros(n_vars, dtype=float)
    for edge in edges:
        compatibility = data.compatibility[edge]
        line = data.lines[edge[1]]
        c[reg_index[edge]] = compatibility.variable_cost_per_unit
        c[ot_index[edge]] = (
            compatibility.variable_cost_per_unit
            + compatibility.processing_hours_per_unit
            * line.overtime_cost_per_hour
            * float(overtime_cost_multiplier)
        )

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    # Demand satisfaction: sum regular + overtime units >= scaled demand.
    for product_id, product in sorted(data.products.items()):
        row = np.zeros(n_vars, dtype=float)
        for edge in edges:
            if edge[0] == product_id:
                row[reg_index[edge]] = 1.0
                row[ot_index[edge]] = 1.0
        rows.append(row)
        lower.append(product.demand_units * float(demand_multiplier))
        upper.append(np.inf)

    # Separate regular and overtime line-capacity constraints.
    for line_id, line in sorted(data.lines.items()):
        regular_row = np.zeros(n_vars, dtype=float)
        overtime_row = np.zeros(n_vars, dtype=float)
        for edge in edges:
            if edge[1] != line_id:
                continue
            processing = data.compatibility[edge].processing_hours_per_unit
            regular_row[reg_index[edge]] = processing
            overtime_row[ot_index[edge]] = processing
        rows.append(regular_row)
        lower.append(-np.inf)
        upper.append(line.regular_capacity_hours)
        rows.append(overtime_row)
        lower.append(-np.inf)
        upper.append(line.overtime_capacity_hours)

    matrix = np.vstack(rows)
    constraint = LinearConstraint(matrix, np.array(lower), np.array(upper))
    bounds = Bounds(np.zeros(n_vars, dtype=float), np.full(n_vars, np.inf, dtype=float))
    result = milp(
        c=c,
        integrality=np.zeros(n_vars, dtype=int),
        bounds=bounds,
        constraints=constraint,
        options={
            "time_limit": float(milp_time),
            "mip_rel_gap": float(mip_gap),
            "presolve": True,
        },
    )

    success = bool(result.success and result.x is not None)
    solution_rows: list[dict[str, object]] = []
    if success:
        x = np.asarray(result.x, dtype=float)
        for edge in edges:
            regular_units = max(0.0, float(x[reg_index[edge]]))
            overtime_units = max(0.0, float(x[ot_index[edge]]))
            units = regular_units + overtime_units
            if units <= 1e-9:
                continue
            solution_rows.append(
                {
                    "product_id": edge[0],
                    "line_id": edge[1],
                    "regular_units": round(regular_units, 10),
                    "overtime_units": round(overtime_units, 10),
                    "units": round(units, 10),
                }
            )

    return {
        "success": success,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective": float(result.fun) if result.fun is not None else None,
        "solution_rows": solution_rows,
        "variable_count": n_vars,
        "constraint_count": len(rows),
    }


__all__ = ["optimize_production_plan"]
