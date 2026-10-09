"""Independent deterministic validator for Production Capacity Planning solutions.

This module intentionally does not import the production optimizer or its feasibility
helpers. It reconstructs feasibility and economics directly from source data and rows.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from production_data import ProductionPlanningData


_TOL = 1e-7


def _number(value: object, field: str, errors: list[str]) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field} must be numeric")
        return None
    if not math.isfinite(number):
        errors.append(f"{field} must be finite")
        return None
    return number


def validate_production_solution(
    rows: Sequence[Mapping[str, object]],
    data: ProductionPlanningData,
    *,
    demand_multiplier: float,
    overtime_cost_multiplier: float,
) -> dict[str, object]:
    errors: list[str] = []
    production_by_product = {product_id: 0.0 for product_id in data.products}
    regular_hours_by_line = {line_id: 0.0 for line_id in data.lines}
    overtime_hours_by_line = {line_id: 0.0 for line_id in data.lines}
    production_cost = 0.0
    overtime_cost = 0.0
    seen_pairs: set[tuple[str, str]] = set()

    for index, row in enumerate(rows):
        product_id = str(row.get("product_id", "")).strip()
        line_id = str(row.get("line_id", "")).strip()
        pair = (product_id, line_id)
        if pair in seen_pairs:
            errors.append(f"duplicate product-line row: {product_id}/{line_id}")
        seen_pairs.add(pair)

        compatibility = data.compatibility.get(pair)
        if compatibility is None:
            errors.append(f"row {index} uses incompatible or unknown product-line pair: {product_id}/{line_id}")
            continue

        regular_units = _number(row.get("regular_units"), f"row {index} regular_units", errors)
        overtime_units = _number(row.get("overtime_units"), f"row {index} overtime_units", errors)
        units = _number(row.get("units"), f"row {index} units", errors)
        if regular_units is None or overtime_units is None or units is None:
            continue
        if regular_units < -_TOL or overtime_units < -_TOL or units < -_TOL:
            errors.append(f"row {index} quantities must be non-negative")
            continue
        if abs((regular_units + overtime_units) - units) > _TOL:
            errors.append(f"row {index} units must equal regular_units + overtime_units")

        production_by_product[product_id] += units
        regular_hours = regular_units * compatibility.processing_hours_per_unit
        overtime_hours = overtime_units * compatibility.processing_hours_per_unit
        regular_hours_by_line[line_id] += regular_hours
        overtime_hours_by_line[line_id] += overtime_hours
        production_cost += units * compatibility.variable_cost_per_unit
        overtime_cost += (
            overtime_hours
            * data.lines[line_id].overtime_cost_per_hour
            * float(overtime_cost_multiplier)
        )

    for product_id, product in data.products.items():
        required = product.demand_units * float(demand_multiplier)
        actual = production_by_product[product_id]
        if actual + _TOL < required:
            errors.append(
                f"demand not satisfied for {product_id}: required {required:.6f}, produced {actual:.6f}"
            )

    for line_id, line in data.lines.items():
        regular = regular_hours_by_line[line_id]
        overtime = overtime_hours_by_line[line_id]
        if regular > line.regular_capacity_hours + _TOL:
            errors.append(
                f"regular capacity exceeded for {line_id}: {regular:.6f} > {line.regular_capacity_hours:.6f}"
            )
        if overtime > line.overtime_capacity_hours + _TOL:
            errors.append(
                f"overtime limit exceeded for {line_id}: {overtime:.6f} > {line.overtime_capacity_hours:.6f}"
            )

    total_cost = production_cost + overtime_cost
    return {
        "valid": not errors,
        "production_cost": round(production_cost, 8),
        "overtime_cost": round(overtime_cost, 8),
        "total_cost": round(total_cost, 8),
        "total_units": round(sum(production_by_product.values()), 8),
        "overtime_hours": round(sum(overtime_hours_by_line.values()), 8),
        "regular_hours": round(sum(regular_hours_by_line.values()), 8),
        "error_count": len(errors),
        "errors": errors,
    }


__all__ = ["validate_production_solution"]
