#!/usr/bin/env python3
"""Carga, limpia y cruza los cuatro CSV originales con pandas.

Esta capa existe para que las decisiones sobre calidad de datos sean visibles antes de
entrar al optimizador. El solver recibe una tabla ya validada y trabaja únicamente con
NumPy y SciPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

PLANTS: Tuple[str, ...] = (
    "buenos_aires",
    "curitiba",
    "santiago",
    "monterrey",
    "bakersfield",
)

CATALOG_COLUMNS = (
    "codigo_producto",
    "caja_tipo_id",
    "peso_neto_caja",
)
SPEC_COLUMNS = (
    "caja_tipo_id",
    "caja_interior_largo",
    "caja_interior_ancho",
    "caja_interior_alto",
)
OPERATION_COLUMNS = (
    "codigo_producto",
    "costo_total",
    "costo_pallets_total",
    *(f"volumen_producto_planta_{plant}" for plant in PLANTS),
)
PROCUREMENT_COLUMNS = (
    "caja_tipo_id",
    "costo_unitario_base",
    *(f"costo_unitario_planta_{plant}" for plant in PLANTS),
    *(f"volumen_tipo_planta_{plant}" for plant in PLANTS),
)


@dataclass(frozen=True)
class PreparedData:
    catalog: pd.DataFrame
    specs: pd.DataFrame
    operations: pd.DataFrame
    procurement: pd.DataFrame
    products: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    # Leer como texto evita que pandas decida tipos antes de tratar comas, ERROR y vacíos.
    frame = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    frame.columns = frame.columns.str.strip()
    return frame


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: faltan columnas requeridas: {missing}")


def _clean_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _to_numeric(series: pd.Series, *, allow_missing: bool = False) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace("mm", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "error": pd.NA})
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if not allow_missing and numeric.isna().any():
        examples = series[numeric.isna()].head(5).tolist()
        raise ValueError(f"Valores numéricos inválidos: {examples}")
    return numeric.astype(float)


def _require_unique(frame: pd.DataFrame, key: str, name: str) -> None:
    duplicated = frame.loc[frame[key].duplicated(keep=False), key].unique().tolist()
    if duplicated:
        raise ValueError(f"{name}: la clave {key} no es única. Ejemplos: {duplicated[:5]}")


def load_inputs(
    catalog_path: Path,
    specs_path: Path,
    operations_path: Path,
    procurement_path: Path,
) -> PreparedData:
    """Prepara una única tabla por producto sin ocultar pérdidas en los cruces."""
    catalog = _read_csv(catalog_path)
    specs = _read_csv(specs_path)
    operations = _read_csv(operations_path)
    procurement = _read_csv(procurement_path)

    _require_columns(catalog, CATALOG_COLUMNS, "catalogo_productos")
    _require_columns(specs, SPEC_COLUMNS, "especificaciones_cajas")
    _require_columns(operations, OPERATION_COLUMNS, "operaciones_planta")
    _require_columns(procurement, PROCUREMENT_COLUMNS, "procurement_cajas")

    for frame, key in (
        (catalog, "codigo_producto"),
        (catalog, "caja_tipo_id"),
        (specs, "caja_tipo_id"),
        (operations, "codigo_producto"),
        (procurement, "caja_tipo_id"),
    ):
        frame[key] = _clean_key(frame[key])

    _require_unique(catalog, "codigo_producto", "catalogo_productos")
    _require_unique(specs, "caja_tipo_id", "especificaciones_cajas")
    _require_unique(operations, "codigo_producto", "operaciones_planta")
    _require_unique(procurement, "caja_tipo_id", "procurement_cajas")

    catalog["peso_neto_caja"] = _to_numeric(catalog["peso_neto_caja"])
    for column in SPEC_COLUMNS[1:]:
        specs[column] = _to_numeric(specs[column])
    for column in OPERATION_COLUMNS[1:]:
        operations[column] = _to_numeric(operations[column])

    procurement["costo_unitario_base"] = _to_numeric(
        procurement["costo_unitario_base"]
    )
    for plant in PLANTS:
        procurement[f"volumen_tipo_planta_{plant}"] = _to_numeric(
            procurement[f"volumen_tipo_planta_{plant}"], allow_missing=True
        )

    # validate= convierte duplicados inesperados en un error, en lugar de multiplicar filas.
    products = (
        catalog.loc[:, CATALOG_COLUMNS]
        .merge(
            specs.loc[:, SPEC_COLUMNS],
            on="caja_tipo_id",
            how="left",
            validate="many_to_one",
            indicator="_spec_match",
        )
        .merge(
            operations.loc[:, OPERATION_COLUMNS],
            on="codigo_producto",
            how="left",
            validate="one_to_one",
            indicator="_operation_match",
        )
    )

    missing_specs = products.loc[
        products["_spec_match"] != "both", "codigo_producto"
    ].tolist()
    missing_operations = products.loc[
        products["_operation_match"] != "both", "codigo_producto"
    ].tolist()
    if missing_specs:
        raise ValueError(f"Productos sin especificación de caja: {missing_specs[:10]}")
    if missing_operations:
        raise ValueError(f"Productos sin datos de operación: {missing_operations[:10]}")

    products = products.drop(columns=["_spec_match", "_operation_match"])
    return PreparedData(
        catalog=catalog,
        specs=specs,
        operations=operations,
        procurement=procurement,
        products=products,
    )


def tier_factor_series(volume: pd.Series) -> pd.Series:
    values = volume.to_numpy(dtype=float)
    factors = np.select(
        [
            values < 20_000,
            values < 50_000,
            values < 100_000,
            values < 500_000,
        ],
        [1.10, 1.00, 0.90, 0.80],
        default=0.70,
    )
    return pd.Series(factors, index=volume.index, dtype=float)


def audit_procurement(data: PreparedData) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Contrasta precios observados con los tiers derivados de Operations.

    Procurement se usa como evidencia de control, no como demanda. Las celdas ERROR no
    se imputan porque la solución futura recalcula cada precio desde volumen y tier.
    """
    product_box = data.catalog[["codigo_producto", "caja_tipo_id"]]
    volume_columns = [f"volumen_producto_planta_{plant}" for plant in PLANTS]

    operations_long = data.operations[["codigo_producto", *volume_columns]].melt(
        id_vars="codigo_producto",
        value_vars=volume_columns,
        var_name="plant_column",
        value_name="operations_volume",
    )
    operations_long["planta"] = operations_long["plant_column"].str.removeprefix(
        "volumen_producto_planta_"
    )
    box_volume = (
        product_box.merge(operations_long, on="codigo_producto", validate="one_to_many")
        .groupby(["caja_tipo_id", "planta"], as_index=False)["operations_volume"]
        .sum()
    )

    price_columns = [f"costo_unitario_planta_{plant}" for plant in PLANTS]
    price_long = data.procurement[
        ["caja_tipo_id", "costo_unitario_base", *price_columns]
    ].melt(
        id_vars=["caja_tipo_id", "costo_unitario_base"],
        value_vars=price_columns,
        var_name="plant_column",
        value_name="observed_raw",
    )
    price_long["planta"] = price_long["plant_column"].str.removeprefix(
        "costo_unitario_planta_"
    )
    price_long["observed_price"] = _to_numeric(
        price_long["observed_raw"], allow_missing=True
    )

    detail = price_long.merge(
        box_volume,
        on=["caja_tipo_id", "planta"],
        how="left",
        validate="one_to_one",
    )
    detail["operations_volume"] = detail["operations_volume"].fillna(0.0)
    detail["expected_factor"] = tier_factor_series(detail["operations_volume"])
    detail["expected_price"] = (
        detail["costo_unitario_base"] * detail["expected_factor"]
    )
    detail["is_error"] = (
        detail["observed_raw"].astype("string").str.strip().str.upper() == "ERROR"
    )
    detail["matches_operations_tier"] = np.isclose(
        detail["observed_price"],
        detail["expected_price"],
        atol=1e-6,
        rtol=0.0,
        equal_nan=False,
    )

    numeric = detail["observed_price"].notna()
    summary: Dict[str, object] = {
        "catalog_rows": int(len(data.catalog)),
        "specification_rows": int(len(data.specs)),
        "operation_rows": int(len(data.operations)),
        "procurement_rows": int(len(data.procurement)),
        "merged_product_rows": int(len(data.products)),
        "procurement_error_cells": int(detail["is_error"].sum()),
        "procurement_numeric_price_cells": int(numeric.sum()),
        "prices_matching_operations_tiers": int(
            detail.loc[numeric, "matches_operations_tier"].sum()
        ),
        "matching_rate": (
            float(detail.loc[numeric, "matches_operations_tier"].mean())
            if numeric.any()
            else None
        ),
        "demand_source": "operaciones_planta",
        "procurement_role": "auditoria; los tiers de la solucion se recalculan",
    }

    columns = [
        "caja_tipo_id",
        "planta",
        "operations_volume",
        "costo_unitario_base",
        "observed_raw",
        "observed_price",
        "expected_factor",
        "expected_price",
        "is_error",
        "matches_operations_tier",
    ]
    return summary, detail.loc[:, columns].sort_values(["caja_tipo_id", "planta"])


def baseline_from_operations(operations: pd.DataFrame) -> Dict[str, float]:
    packaging = float(operations["costo_total"].sum())
    freight = float(operations["costo_pallets_total"].sum())
    return {"packaging": packaging, "freight": freight, "total": packaging + freight}
