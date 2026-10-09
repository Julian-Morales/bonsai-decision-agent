#!/usr/bin/env python3
"""Validador independiente para submissions de Bonsai Corp.

Modos:
- faq: permite variación ±10% por eje, volumen interno >= original y no permuta ejes.
- strict: además exige que ninguna dimensión interna sea menor que la original.

El validador usa Operations como fuente de demanda y USD 150/pallet como escenario
base intrarregional, en línea con los archivos suministrados.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

PLANTS = ("buenos_aires", "curitiba", "santiago", "monterrey", "bakersfield")
PALLET_LARGO = 1200.0
PALLET_ANCHO = 800.0
PALLET_ALTO = 1800.0
G = 9.81
ALLOWED = {
    3.0: {"ect": 1000.0, "price": 0.60, "headspace": 0.06},
    4.5: {"ect": 1400.0, "price": 0.65, "headspace": 0.08},
    5.0: {"ect": 1650.0, "price": 0.70, "headspace": 0.10},
}
EPS = 1e-5


def parse_float(x: object) -> float:
    return float(str(x).strip().lower().replace("mm", "").replace(" ", "").replace(",", "."))


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def tier_factor(volume: float) -> float:
    if volume < 20_000:
        return 1.10
    if volume < 50_000:
        return 1.00
    if volume < 100_000:
        return 0.90
    if volume < 500_000:
        return 0.80
    return 0.70


def _validate_rows(
    submission: Sequence[Mapping[str, object]],
    catalog_path: Path,
    specs_path: Path,
    operations_path: Path,
    mode: str,
    pallet_cost: float,
    *,
    submission_label: str | None,
    actual_columns: Sequence[str] | None,
) -> Dict[str, object]:
    """Shared validator-owned semantic engine for file and in-memory rows."""
    catalog = read_rows(catalog_path)
    specs = {r["caja_tipo_id"]: r for r in read_rows(specs_path)}
    operations = {r["codigo_producto"]: r for r in read_rows(operations_path)}
    rows = [dict(row) for row in submission]

    errors: List[Dict[str, str]] = []
    expected_columns = [
        "codigo_producto",
        "caja_grosor_mm",
        "caja_exterior_largo",
        "caja_exterior_ancho",
        "caja_exterior_alto",
    ]
    if actual_columns is not None and list(actual_columns) != expected_columns:
        errors.append({"rule": "OUT-04", "product": "", "detail": f"Columnas: {list(actual_columns)}"})

    codes = [str(r.get("codigo_producto", "")) for r in rows]
    expected_codes = [r["codigo_producto"] for r in catalog]
    if len(codes) != len(set(codes)):
        errors.append({"rule": "OUT-03", "product": "", "detail": "Hay códigos duplicados"})
    missing = sorted(set(expected_codes) - set(codes))
    extra = sorted(set(codes) - set(expected_codes))
    if missing:
        errors.append({"rule": "OUT-02", "product": "", "detail": f"Faltantes: {missing[:10]}"})
    if extra:
        errors.append({"rule": "OUT-01", "product": "", "detail": f"Códigos desconocidos: {extra[:10]}"})

    submission_by_code = {str(r["codigo_producto"]): r for r in rows if r.get("codigo_producto")}
    thicknesses = set()
    type_volumes: Dict[Tuple[float, float, float, float], List[float]] = {}
    freight = 0.0
    q_by_code: Dict[str, int] = {}
    min_fit_margin = float("inf")
    min_volume_margin = float("inf")
    min_ect_margin = float("inf")
    max_headspace_excess = -float("inf")

    for item in catalog:
        code = item["codigo_producto"]
        if code not in submission_by_code or code not in operations:
            continue
        row = submission_by_code[code]
        try:
            thickness = parse_float(row["caja_grosor_mm"])
            ext = [
                parse_float(row["caja_exterior_largo"]),
                parse_float(row["caja_exterior_ancho"]),
                parse_float(row["caja_exterior_alto"]),
            ]
        except Exception as exc:
            errors.append({"rule": "NUM", "product": code, "detail": str(exc)})
            continue
        thicknesses.add(round(thickness, 6))
        if thickness not in ALLOWED:
            errors.append({"rule": "GRO-01", "product": code, "detail": f"Grosor {thickness}"})
            continue
        cfg = ALLOWED[thickness]
        internal = [x - 2.0 * thickness for x in ext]
        if min(internal) <= 0:
            errors.append({"rule": "DIM-03", "product": code, "detail": "Dimensión interna no positiva"})
            continue

        spec = specs[item["caja_tipo_id"]]
        orig = [
            parse_float(spec["caja_interior_largo"]),
            parse_float(spec["caja_interior_ancho"]),
            parse_float(spec["caja_interior_alto"]),
        ]

        for axis, (new, old) in enumerate(zip(internal, orig)):
            lower = 0.90 * old
            upper = 1.10 * old
            min_fit_margin = min(min_fit_margin, new - lower, upper - new)
            if new < lower - EPS or new > upper + EPS:
                errors.append({"rule": "FIT-01", "product": code, "detail": f"Eje {axis}: {new} vs {old}"})
            if mode == "strict" and new < old - EPS:
                errors.append({"rule": "MIN-01", "product": code, "detail": f"Eje {axis}: {new} < {old}"})

            headspace = new - old
            maximum = min(cfg["headspace"] * new, 40.0)
            max_headspace_excess = max(max_headspace_excess, headspace - maximum)
            if headspace > maximum + EPS:
                errors.append({"rule": "HSP", "product": code, "detail": f"Eje {axis}: {headspace} > {maximum}"})

        original_volume = orig[0] * orig[1] * orig[2]
        new_volume = internal[0] * internal[1] * internal[2]
        min_volume_margin = min(min_volume_margin, new_volume - original_volume)
        if new_volume + EPS < original_volume:
            errors.append({"rule": "FIT-04", "product": code, "detail": f"{new_volume} < {original_volume}"})

        n_l = math.floor((PALLET_ANCHO + 1e-9) / ext[0])
        n_w = math.floor((PALLET_LARGO + 1e-9) / ext[1])
        n_h = math.floor((PALLET_ALTO + 1e-9) / ext[2])
        q = n_l * n_w * n_h
        q_by_code[code] = q
        if q < 1:
            errors.append({"rule": "PAL-07", "product": code, "detail": f"q={q}"})
            continue

        load_max = cfg["ect"] * (2.0 * (ext[0] + ext[1]) / 1000.0) / G
        required = max(0, n_h - 1) * parse_float(item["peso_neto_caja"])
        min_ect_margin = min(min_ect_margin, load_max - required)
        if load_max + EPS < required:
            errors.append({"rule": "ECT-04", "product": code, "detail": f"{load_max} < {required}"})

        op = operations[code]
        volumes = [parse_float(op[f"volumen_producto_planta_{p}"]) for p in PLANTS]
        freight += sum(math.ceil(v / q) * pallet_cost if v > 0 else 0.0 for v in volumes)
        key = (round(thickness, 6), round(ext[0], 6), round(ext[1], 6), round(ext[2], 6))
        accumulator = type_volumes.setdefault(key, [0.0] * len(PLANTS))
        for p, volume in enumerate(volumes):
            accumulator[p] += volume

    if len(thicknesses) != 1:
        errors.append({"rule": "GRO-02", "product": "", "detail": f"Grosores: {sorted(thicknesses)}"})

    packaging = 0.0
    for key, volumes in type_volumes.items():
        thickness = key[0]
        price = ALLOWED[thickness]["price"]
        for volume in volumes:
            if volume > 0:
                packaging += volume * price * tier_factor(volume)

    operations_rows = list(operations.values())
    baseline_packaging = sum(parse_float(r["costo_total"]) for r in operations_rows)
    baseline_freight = sum(parse_float(r["costo_pallets_total"]) for r in operations_rows)
    baseline_total = baseline_packaging + baseline_freight
    total = packaging + freight

    return {
        "submission": submission_label,
        "mode": mode,
        "valid": not errors,
        "n_products_expected": len(catalog),
        "n_products_submission": len(rows),
        "n_types": len(type_volumes),
        "thicknesses": sorted(thicknesses),
        "packaging": packaging,
        "freight": freight,
        "total": total,
        "baseline_packaging": baseline_packaging,
        "baseline_freight": baseline_freight,
        "baseline_total": baseline_total,
        "saving": baseline_total - total,
        "saving_pct": 100.0 * (baseline_total - total) / baseline_total,
        "minimum_fit_margin_mm": min_fit_margin,
        "minimum_volume_margin_mm3": min_volume_margin,
        "minimum_ect_margin_kg": min_ect_margin,
        "maximum_headspace_excess_mm": max_headspace_excess,
        "error_count": len(errors),
        "errors": errors,
    }


def validate_solution_rows(
    rows: Sequence[Mapping[str, object]],
    catalog_path: Path,
    specs_path: Path,
    operations_path: Path,
    mode: str = "faq",
    pallet_cost: float = 150.0,
) -> Dict[str, object]:
    """Validate generic in-memory solution mappings with the independent validator."""
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Cada fila de solución debe ser un Mapping")
        normalized.append(dict(row))
    return _validate_rows(
        normalized,
        catalog_path,
        specs_path,
        operations_path,
        mode,
        pallet_cost,
        submission_label=None,
        actual_columns=None,
    )


def validate(
    submission_path: Path,
    catalog_path: Path,
    specs_path: Path,
    operations_path: Path,
    mode: str,
    pallet_cost: float,
) -> Dict[str, object]:
    """Validate the persisted CSV artifact, including its exact output schema."""
    submission = read_rows(submission_path)
    actual_columns = list(submission[0].keys()) if submission else []
    return _validate_rows(
        submission,
        catalog_path,
        specs_path,
        operations_path,
        mode,
        pallet_cost,
        submission_label=str(submission_path),
        actual_columns=actual_columns,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/data"))
    parser.add_argument("--catalog", default="catalogo_productos(1).csv")
    parser.add_argument("--specs", default="especificaciones_cajas(2).csv")
    parser.add_argument("--operations", default="operaciones_planta(2).csv")
    parser.add_argument("--mode", choices=("faq", "strict"), default="faq")
    parser.add_argument("--pallet-cost", type=float, default=150.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = validate(
        args.submission,
        args.data_dir / args.catalog,
        args.data_dir / args.specs,
        args.data_dir / args.operations,
        mode=args.mode,
        pallet_cost=args.pallet_cost,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    raise SystemExit(0 if report["valid"] else 2)


if __name__ == "__main__":
    main()
