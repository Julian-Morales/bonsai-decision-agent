"""Deterministic bundled data loader for the Production Capacity Planning demo domain."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class Product:
    product_id: str
    demand_units: float


@dataclass(frozen=True)
class ProductionLine:
    line_id: str
    regular_capacity_hours: float
    overtime_capacity_hours: float
    overtime_cost_per_hour: float


@dataclass(frozen=True)
class Compatibility:
    product_id: str
    line_id: str
    processing_hours_per_unit: float
    variable_cost_per_unit: float


@dataclass(frozen=True)
class ProductionPlanningData:
    products: Mapping[str, Product]
    lines: Mapping[str, ProductionLine]
    compatibility: Mapping[tuple[str, str], Compatibility]
    current_plan: Tuple[Mapping[str, object], ...]


DEFAULT_PRODUCTION_DATA_DIR = Path(__file__).resolve().parent / "production_demo"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def load_production_data(data_dir: Path | None = None) -> ProductionPlanningData:
    root = Path(data_dir or DEFAULT_PRODUCTION_DATA_DIR)
    product_rows = _read_csv(root / "products.csv")
    line_rows = _read_csv(root / "lines.csv")
    compatibility_rows = _read_csv(root / "compatibility.csv")
    current_plan_rows = _read_csv(root / "current_plan.csv")

    products: Dict[str, Product] = {}
    for row in product_rows:
        product_id = str(row["product_id"]).strip()
        if not product_id or product_id in products:
            raise ValueError(f"Invalid or duplicate product_id: {product_id!r}")
        demand = float(row["demand_units"])
        if demand <= 0:
            raise ValueError(f"Product demand must be positive: {product_id}")
        products[product_id] = Product(product_id=product_id, demand_units=demand)

    lines: Dict[str, ProductionLine] = {}
    for row in line_rows:
        line_id = str(row["line_id"]).strip()
        if not line_id or line_id in lines:
            raise ValueError(f"Invalid or duplicate line_id: {line_id!r}")
        regular = float(row["regular_capacity_hours"])
        overtime = float(row["overtime_capacity_hours"])
        overtime_cost = float(row["overtime_cost_per_hour"])
        if regular < 0 or overtime < 0 or overtime_cost < 0:
            raise ValueError(f"Line capacities/costs must be non-negative: {line_id}")
        lines[line_id] = ProductionLine(
            line_id=line_id,
            regular_capacity_hours=regular,
            overtime_capacity_hours=overtime,
            overtime_cost_per_hour=overtime_cost,
        )

    compatibility: Dict[tuple[str, str], Compatibility] = {}
    for row in compatibility_rows:
        product_id = str(row["product_id"]).strip()
        line_id = str(row["line_id"]).strip()
        key = (product_id, line_id)
        if product_id not in products or line_id not in lines:
            raise ValueError(f"Compatibility references unknown product/line: {key}")
        if key in compatibility:
            raise ValueError(f"Duplicate compatibility row: {key}")
        processing = float(row["processing_hours_per_unit"])
        variable_cost = float(row["variable_cost_per_unit"])
        if processing <= 0 or variable_cost < 0:
            raise ValueError(f"Invalid compatibility economics: {key}")
        compatibility[key] = Compatibility(
            product_id=product_id,
            line_id=line_id,
            processing_hours_per_unit=processing,
            variable_cost_per_unit=variable_cost,
        )

    for product_id in products:
        if not any(key[0] == product_id for key in compatibility):
            raise ValueError(f"Product has no compatible production line: {product_id}")

    current_plan: list[dict[str, object]] = []
    for row in current_plan_rows:
        current_plan.append(
            {
                "product_id": str(row["product_id"]).strip(),
                "line_id": str(row["line_id"]).strip(),
                "regular_units": float(row["regular_units"]),
                "overtime_units": float(row["overtime_units"]),
                "units": float(row["units"]),
            }
        )

    return ProductionPlanningData(
        products=products,
        lines=lines,
        compatibility=compatibility,
        current_plan=tuple(current_plan),
    )


def audit_production_data(data: ProductionPlanningData) -> dict[str, object]:
    """Return compact structural checks without calling optimizer logic."""
    incompatible_current_rows = 0
    for row in data.current_plan:
        if (str(row["product_id"]), str(row["line_id"])) not in data.compatibility:
            incompatible_current_rows += 1
    return {
        "ok": incompatible_current_rows == 0,
        "product_count": len(data.products),
        "line_count": len(data.lines),
        "compatibility_count": len(data.compatibility),
        "current_plan_row_count": len(data.current_plan),
        "incompatible_current_plan_rows": incompatible_current_rows,
    }


__all__ = [
    "Compatibility",
    "DEFAULT_PRODUCTION_DATA_DIR",
    "Product",
    "ProductionLine",
    "ProductionPlanningData",
    "audit_production_data",
    "load_production_data",
]
