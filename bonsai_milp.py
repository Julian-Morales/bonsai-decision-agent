#!/usr/bin/env python3
"""
Refinamiento MILP del portafolio de cajas de Bonsai Corp.

La geometría no permuta ejes: largo->largo, ancho->ancho y alto->alto.
La variante principal aplica literalmente el FAQ corregido:
- cada eje interno puede variar ±10%;
- el volumen interno no puede ser menor que el volumen interno original;
- headspace máximo por eje; una reducción de eje no genera headspace positivo.

El modelo toma como catálogo semilla las cajas producidas por bonsai_core.py,
agrega diseños singleton prometedores y optimiza globalmente la asignación de SKU a
tipos de caja. Los tiers de packaging se modelan por tipo y planta mediante variables
binarias de banda. El flete se calcula SKU por SKU, sin mezclar productos en un pallet.

Dependencias: numpy, scipy y bonsai_core.py.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

# Carga robusta del módulo base desde la misma carpeta del script.
BASE_MODULE_PATH = Path(__file__).with_name("bonsai_core.py")
spec = importlib.util.spec_from_file_location("bonsai_core", BASE_MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"No se pudo cargar {BASE_MODULE_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

PLANTS = base.PLANTS
P = len(PLANTS)
K = 5
THICKNESS = 3.0
ECT = 1000.0
BASE_PRICE = 0.60
HEADSPACE_RATIO = 0.06
PALLET_COST = 150.0
EPS = 1e-6
TIER_FACTORS = (1.10, 1.00, 0.90, 0.80, 0.70)
TIER_LOWERS = (1.0, 20_000.0, 50_000.0, 100_000.0, 500_000.0)
TIER_UPPERS_FIXED = (19_999.0, 49_999.0, 99_999.0, 499_999.0)


@dataclass(frozen=True)
class CandidateBox:
    dims_internal: Tuple[float, float, float]
    q: int
    source: str
    thickness: float = THICKNESS

    @property
    def dims_external(self) -> Tuple[float, float, float]:
        return tuple(x + 2.0 * self.thickness for x in self.dims_internal)

    @property
    def key(self) -> Tuple[float, float, float]:
        return tuple(round(x, 6) for x in self.dims_internal)


@dataclass(frozen=True)
class SubmissionRow:
    codigo_producto: str
    caja_grosor_mm: float
    caja_exterior_largo: float
    caja_exterior_ancho: float
    caja_exterior_alto: float


def pallet_capacity(dims_internal: Sequence[float], thickness: float = THICKNESS) -> int:
    l, w, h = (float(x) + 2.0 * thickness for x in dims_internal)
    return (
        math.floor((base.PALLET_ANCHO + 1e-9) / l)
        * math.floor((base.PALLET_LARGO + 1e-9) / w)
        * math.floor((base.PALLET_ALTO + 1e-9) / h)
    )


def is_compatible(
    product: base.Product,
    candidate: CandidateBox,
    *,
    thickness: float | None = None,
    ect: float = ECT,
    headspace_ratio: float = HEADSPACE_RATIO,
) -> bool:
    """Compatibilidad FAQ corregido, sin permutación de ejes."""
    dims = np.asarray(candidate.dims_internal, dtype=float)
    orig = np.asarray(product.orig_dims, dtype=float)
    thickness = candidate.thickness if thickness is None else float(thickness)

    lo = 0.90 * orig
    hi = np.minimum.reduce([1.10 * orig, orig / (1.0 - headspace_ratio), orig + 40.0])
    if np.any(dims < lo - EPS) or np.any(dims > hi + EPS):
        return False
    if float(np.prod(dims)) + 1e-5 < product.orig_volume:
        return False

    headspace = dims - orig
    max_headspace = np.minimum(headspace_ratio * dims, 40.0)
    if np.any(headspace > max_headspace + EPS):
        return False

    ext = dims + 2.0 * thickness
    n_h = math.floor((base.PALLET_ALTO + 1e-9) / ext[2])
    load_max = ect * (2.0 * (ext[0] + ext[1]) / 1000.0) / base.G
    required = max(0, n_h - 1) * product.weight_kg
    return load_max + EPS >= required


def freight_cost(product: base.Product, q: int, pallet_cost: float = PALLET_COST) -> float:
    return sum(base.ceil_div_volume(v, q) * pallet_cost for v in product.volumes)


def seed_candidates_from_rows(
    rows: Sequence[SubmissionRow],
    *,
    thickness: float = THICKNESS,
) -> Tuple[List[CandidateBox], Dict[str, Tuple[float, float, float]]]:
    """Construye el catálogo semilla directamente desde filas de solución en memoria."""
    candidates: Dict[Tuple[float, float, float], CandidateBox] = {}
    assignment_keys: Dict[str, Tuple[float, float, float]] = {}
    for row in rows:
        row_thickness = float(row.caja_grosor_mm)
        if abs(row_thickness - float(thickness)) > 1e-9:
            raise ValueError(f"La semilla debe utilizar grosor uniforme de {thickness:.1f} mm")
        dims = (
            float(row.caja_exterior_largo) - 2.0 * thickness,
            float(row.caja_exterior_ancho) - 2.0 * thickness,
            float(row.caja_exterior_alto) - 2.0 * thickness,
        )
        key = tuple(round(x, 6) for x in dims)
        candidates.setdefault(key, CandidateBox(dims, pallet_capacity(dims, thickness), "seed", thickness))
        assignment_keys[row.codigo_producto] = key
    return list(candidates.values()), assignment_keys


def read_seed_candidates(
    seed_submission: Path,
) -> Tuple[List[CandidateBox], Dict[str, Tuple[float, float, float]]]:
    """Legacy file wrapper around :func:`seed_candidates_from_rows`."""
    with seed_submission.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = [
            SubmissionRow(
                codigo_producto=row["codigo_producto"],
                caja_grosor_mm=float(row["caja_grosor_mm"]),
                caja_exterior_largo=float(row["caja_exterior_largo"]),
                caja_exterior_ancho=float(row["caja_exterior_ancho"]),
                caja_exterior_alto=float(row["caja_exterior_alto"]),
            )
            for row in csv.DictReader(fh)
        ]
    return seed_candidates_from_rows(rows)

def build_candidate_pool(
    products: Sequence[base.Product],
    seed_candidates: Sequence[CandidateBox],
    seed_assignment_keys: Dict[str, Tuple[float, float, float]],
    extra_candidates: int,
    *,
    thickness: float = THICKNESS,
    pallet_cost: float = PALLET_COST,
    ect: float = ECT,
    headspace_ratio: float = HEADSPACE_RATIO,
    base_price: float = BASE_PRICE,
) -> List[CandidateBox]:
    pool: Dict[Tuple[float, float, float], CandidateBox] = {c.key: c for c in seed_candidates}
    seed_by_key = {c.key: c for c in seed_candidates}
    seed_q = {p.idx: seed_by_key[seed_assignment_keys[p.code]].q for p in products}

    singleton_optimizer = base.BonsaiOptimizer(products, thickness, pallet_costs=pallet_cost)
    singleton_clusters = singleton_optimizer.singleton_clusters()

    extras: Dict[Tuple[float, float, float], CandidateBox] = {}
    for cluster in singleton_clusters.values():
        dims = tuple(float(x) for x in cluster.dims)
        candidate = CandidateBox(dims, pallet_capacity(dims, thickness), "singleton", thickness)
        if candidate.key not in pool:
            extras.setdefault(candidate.key, candidate)

    ranked = []
    for candidate in extras.values():
        ranking_candidate = CandidateBox(candidate.key, pallet_capacity(candidate.key, thickness), candidate.source, thickness)
        compatible = [p for p in products if is_compatible(p, ranking_candidate, thickness=thickness, ect=ect, headspace_ratio=headspace_ratio)]
        no_freight_penalty = [p for p in compatible if candidate.q >= seed_q[p.idx]]
        volume = sum(float(p.volumes.sum()) for p in no_freight_penalty)
        ranked.append(
            (
                len(no_freight_penalty),
                volume,
                candidate.q,
                len(compatible),
                candidate,
            )
        )
    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

    for *_, candidate in ranked[: max(0, extra_candidates)]:
        pool.setdefault(candidate.key, candidate)
    return list(pool.values())


def solve_assignment_milp(
    products: Sequence[base.Product],
    candidates: Sequence[CandidateBox],
    time_limit: float,
    mip_gap: float,
    *,
    pallet_cost: float = PALLET_COST,
    base_price: float = BASE_PRICE,
    thickness: float = THICKNESS,
    ect: float = ECT,
    headspace_ratio: float = HEADSPACE_RATIO,
):
    edges: List[Tuple[int, int]] = []
    edge_costs: List[float] = []
    compatible_edges: List[List[int]] = [[] for _ in products]
    edges_by_box: List[List[Tuple[int, int]]] = [[] for _ in candidates]

    for i, product in enumerate(products):
        for b, candidate in enumerate(candidates):
            if not is_compatible(product, candidate, thickness=thickness, ect=ect, headspace_ratio=headspace_ratio):
                continue
            edge = len(edges)
            edges.append((i, b))
            edge_costs.append(freight_cost(product, candidate.q, pallet_cost))
            compatible_edges[i].append(edge)
            edges_by_box[b].append((edge, i))

    missing = [products[i].code for i, lst in enumerate(compatible_edges) if not lst]
    if missing:
        raise RuntimeError(f"Productos sin caja candidata: {missing[:10]}")

    n_edges = len(edges)
    n_boxes = len(candidates)
    v_start = n_edges
    y_start = v_start + n_boxes * P * K
    n_vars = y_start + n_boxes * P * K

    objective = np.zeros(n_vars, dtype=float)
    objective[:n_edges] = edge_costs
    for b in range(n_boxes):
        for p in range(P):
            for k, factor in enumerate(TIER_FACTORS):
                objective[v_start + (b * P + p) * K + k] = base_price * factor

    integrality = np.zeros(n_vars, dtype=int)
    integrality[:n_edges] = 1
    integrality[y_start:] = 1

    lower_bounds = np.zeros(n_vars, dtype=float)
    upper_bounds = np.full(n_vars, np.inf, dtype=float)
    upper_bounds[:n_edges] = 1.0
    upper_bounds[y_start:] = 1.0

    row_idx: List[int] = []
    col_idx: List[int] = []
    values: List[float] = []
    constraint_lower: List[float] = []
    constraint_upper: List[float] = []
    row = 0

    def add_coef(column: int, value: float) -> None:
        row_idx.append(row)
        col_idx.append(column)
        values.append(value)

    # Cada producto se asigna exactamente a una caja.
    for edge_list in compatible_edges:
        for edge in edge_list:
            add_coef(edge, 1.0)
        constraint_lower.append(1.0)
        constraint_upper.append(1.0)
        row += 1

    plant_totals = np.sum([p.volumes for p in products], axis=0)
    for b in range(n_boxes):
        for p in range(P):
            # Volumen asignado al tipo/planta = volumen ubicado en una única banda.
            for edge, i in edges_by_box[b]:
                volume = float(products[i].volumes[p])
                if volume:
                    add_coef(edge, volume)
            for k in range(K):
                add_coef(v_start + (b * P + p) * K + k, -1.0)
            constraint_lower.append(0.0)
            constraint_upper.append(0.0)
            row += 1

            tier_uppers = (*TIER_UPPERS_FIXED, max(500_000.0, float(plant_totals[p])))
            for k in range(K):
                v_idx = v_start + (b * P + p) * K + k
                y_idx = y_start + (b * P + p) * K + k

                # v >= lower * y
                add_coef(v_idx, 1.0)
                add_coef(y_idx, -TIER_LOWERS[k])
                constraint_lower.append(0.0)
                constraint_upper.append(np.inf)
                row += 1

                # v <= upper * y
                add_coef(v_idx, 1.0)
                add_coef(y_idx, -tier_uppers[k])
                constraint_lower.append(-np.inf)
                constraint_upper.append(0.0)
                row += 1

            # Como máximo una banda por tipo/planta. Si volumen=0, ninguna banda.
            for k in range(K):
                add_coef(y_start + (b * P + p) * K + k, 1.0)
            constraint_lower.append(-np.inf)
            constraint_upper.append(1.0)
            row += 1

    matrix = coo_matrix((values, (row_idx, col_idx)), shape=(row, n_vars)).tocsr()
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix, np.asarray(constraint_lower), np.asarray(constraint_upper)),
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": float(mip_gap),
            "presolve": True,
            "disp": False,
        },
    )
    if result.x is None:
        raise RuntimeError(f"El MILP no produjo solución: {result.message}")

    assignment: Dict[int, int] = {}
    for edge, value in enumerate(result.x[:n_edges]):
        if value > 0.5:
            i, b = edges[edge]
            assignment[i] = b
    if len(assignment) != len(products):
        raise RuntimeError("La solución MILP no asignó todos los productos")

    return assignment, result, len(edges), n_vars, row


def exact_costs(
    products: Sequence[base.Product],
    candidates: Sequence[CandidateBox],
    assignment: Dict[int, int],
    *,
    base_price: float = BASE_PRICE,
    pallet_cost: float = PALLET_COST,
) -> Tuple[float, float, float]:
    packaging = 0.0
    for b in sorted(set(assignment.values())):
        volumes = np.sum([products[i].volumes for i, bb in assignment.items() if bb == b], axis=0)
        packaging += base.packaging_cost(volumes, base_price)
    freight = sum(freight_cost(products[i], candidates[b].q, pallet_cost) for i, b in assignment.items())
    return packaging, freight, packaging + freight


def submission_rows(
    products: Sequence[base.Product],
    candidates: Sequence[CandidateBox],
    assignment: Dict[int, int],
) -> List[SubmissionRow]:
    """Materializa una asignación como filas de submission, preservando 9 decimales legacy."""
    rows: List[SubmissionRow] = []
    for product in products:
        candidate = candidates[assignment[product.idx]]
        l, w, h = candidate.dims_external
        rows.append(
            SubmissionRow(
                codigo_producto=product.code,
                caja_grosor_mm=candidate.thickness,
                caja_exterior_largo=float(f"{l:.9f}"),
                caja_exterior_ancho=float(f"{w:.9f}"),
                caja_exterior_alto=float(f"{h:.9f}"),
            )
        )
    return rows


def write_submission_rows(path: Path, rows: Sequence[SubmissionRow]) -> None:
    """Serializa filas de solución al formato CSV de Kaggle."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "codigo_producto",
                "caja_grosor_mm",
                "caja_exterior_largo",
                "caja_exterior_ancho",
                "caja_exterior_alto",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "codigo_producto": row.codigo_producto,
                    "caja_grosor_mm": f"{row.caja_grosor_mm:.1f}",
                    "caja_exterior_largo": f"{row.caja_exterior_largo:.9f}",
                    "caja_exterior_ancho": f"{row.caja_exterior_ancho:.9f}",
                    "caja_exterior_alto": f"{row.caja_exterior_alto:.9f}",
                }
            )


def write_submission(
    path: Path,
    products: Sequence[base.Product],
    candidates: Sequence[CandidateBox],
    assignment: Dict[int, int],
) -> None:
    """Legacy convenience wrapper that materializes and serializes an assignment."""
    write_submission_rows(path, submission_rows(products, candidates, assignment))

def write_catalogue(
    path: Path,
    products: Sequence[base.Product],
    candidates: Sequence[CandidateBox],
    assignment: Dict[int, int],
) -> None:
    used = sorted(set(assignment.values()), key=lambda b: (-sum(bb == b for bb in assignment.values()), b))
    with path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "tipo_propuesto",
            "fuente_candidata",
            "n_productos",
            "productos",
            "grosor_mm",
            "interior_largo",
            "interior_ancho",
            "interior_alto",
            "exterior_largo",
            "exterior_ancho",
            "exterior_alto",
            "cajas_por_pallet",
            "volumen_total",
            "costo_packaging",
            "costo_flete",
            "costo_total",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for number, b in enumerate(used, start=1):
            member_indices = sorted(i for i, bb in assignment.items() if bb == b)
            candidate = candidates[b]
            volumes = np.sum([products[i].volumes for i in member_indices], axis=0)
            pack = base.packaging_cost(volumes, base_price)
            freight = sum(freight_cost(products[i], candidate.q, pallet_cost) for i in member_indices)
            li, wi, hi = candidate.dims_internal
            le, we, he = candidate.dims_external
            writer.writerow(
                {
                    "tipo_propuesto": f"BOX_{number:03d}",
                    "fuente_candidata": candidate.source,
                    "n_productos": len(member_indices),
                    "productos": "|".join(products[i].code for i in member_indices),
                    "grosor_mm": f"{candidate.thickness:.1f}",
                    "interior_largo": f"{li:.9f}",
                    "interior_ancho": f"{wi:.9f}",
                    "interior_alto": f"{hi:.9f}",
                    "exterior_largo": f"{le:.9f}",
                    "exterior_ancho": f"{we:.9f}",
                    "exterior_alto": f"{he:.9f}",
                    "cajas_por_pallet": candidate.q,
                    "volumen_total": f"{float(volumes.sum()):.0f}",
                    "costo_packaging": f"{pack:.2f}",
                    "costo_flete": f"{freight:.2f}",
                    "costo_total": f"{pack + freight:.2f}",
                }
            )


def run_scenario(
    name: str,
    products: Sequence[base.Product],
    seed_candidates: Sequence[CandidateBox],
    seed_assignment_keys: Dict[str, Tuple[float, float, float]],
    extra_candidates: int,
    time_limit: float,
    mip_gap: float,
    output_dir: Path,
    baseline: Dict[str, float],
) -> Dict[str, object]:
    candidates = build_candidate_pool(products, seed_candidates, seed_assignment_keys, extra_candidates)
    assignment, result, n_edges, n_vars, n_constraints = solve_assignment_milp(
        products, candidates, time_limit=time_limit, mip_gap=mip_gap
    )
    packaging, freight, total = exact_costs(products, candidates, assignment)

    submission_path = output_dir / f"submission_{name}.csv"
    catalogue_path = output_dir / f"catalogo_{name}.csv"
    write_submission(submission_path, products, candidates, assignment)
    write_catalogue(catalogue_path, products, candidates, assignment)

    return {
        "scenario": name,
        "extra_candidates": extra_candidates,
        "candidate_types": len(candidates),
        "compatibility_edges": n_edges,
        "model_variables": n_vars,
        "model_constraints": n_constraints,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "mip_gap": float(getattr(result, "mip_gap", float("nan"))),
        "n_types": len(set(assignment.values())),
        "packaging": packaging,
        "freight": freight,
        "total": total,
        "saving": baseline["total"] - total,
        "saving_pct": 100.0 * (baseline["total"] - total) / baseline["total"],
        "submission": str(submission_path),
        "catalogue": str(catalogue_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/data"))
    parser.add_argument("--catalog", default="catalogo_productos(1).csv")
    parser.add_argument("--specs", default="especificaciones_cajas(2).csv")
    parser.add_argument("--operations", default="operaciones_planta(2).csv")
    parser.add_argument(
        "--seed-submission",
        type=Path,
        default=Path("/mnt/data/resultados_bonsai/submission_t3_0.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/data/resultados_bonsai_milp"))
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--mip-gap", type=float, default=5e-4)
    parser.add_argument("--expanded-extra-candidates", type=int, default=40)
    args = parser.parse_args()

    products, _specs, operations_rows = base.load_products(
        args.data_dir / args.catalog,
        args.data_dir / args.specs,
        args.data_dir / args.operations,
    )
    baseline = base.official_baseline(operations_rows)
    seed_candidates, seed_assignment_keys = read_seed_candidates(args.seed_submission)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        run_scenario(
            "faq_compact",
            products,
            seed_candidates,
            seed_assignment_keys,
            extra_candidates=0,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            output_dir=args.output_dir,
            baseline=baseline,
        ),
        run_scenario(
            "faq_cost_min",
            products,
            seed_candidates,
            seed_assignment_keys,
            extra_candidates=args.expanded_extra_candidates,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            output_dir=args.output_dir,
            baseline=baseline,
        ),
    ]

    fields = [
        "scenario",
        "extra_candidates",
        "candidate_types",
        "compatibility_edges",
        "model_variables",
        "model_constraints",
        "solver_status",
        "solver_message",
        "mip_gap",
        "n_types",
        "packaging",
        "freight",
        "total",
        "saving",
        "saving_pct",
        "submission",
        "catalogue",
    ]
    with (args.output_dir / "resumen_milp.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    with (args.output_dir / "resumen_milp.json").open("w", encoding="utf-8") as fh:
        json.dump({"baseline": baseline, "scenarios": summaries}, fh, indent=2, ensure_ascii=False)

    print(json.dumps({"baseline": baseline, "scenarios": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
