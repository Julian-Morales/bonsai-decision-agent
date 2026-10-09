#!/usr/bin/env python3
"""Ejecuta la optimización completa de Bonsai Corp desde los cuatro CSV originales.

La búsqueda se divide en etapas porque el modelo global con todas las cajas posibles es
innecesariamente grande. Primero obtiene una solución factible simple; después usa MILP
en subconjuntos relevantes para capturar descuentos por volumen que una búsqueda greedy
no puede detectar.

Salida principal:
    outputs/submission_optimizada.csv

Dependencias:
    numpy, scipy
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import bonsai_core as core
import bonsai_data
import bonsai_milp as model
from validate_submission import validate
from optimization_problem import OptimizationProblem, default_bonsai_packaging_problem


THICKNESS = 3.0
PALLET_COST = 150.0
PLANTS = core.PLANTS
N_PLANTS = len(PLANTS)
N_TIERS = len(model.TIER_FACTORS)


@dataclass(frozen=True)
class OptimizationConfig:
    milp_time: float = 30.0
    mip_gap: float = 5e-4
    extra_candidates: int = 40
    lns_time: float = 0.2
    lns_rounds: int = 2
    target_count: int = 40
    target_time: float = 0.05
    quiet: bool = False
    fast: bool = False


@dataclass(frozen=True)
class OptimizationResult:
    success: bool
    config: OptimizationConfig
    problem: OptimizationProblem
    data_audit: Dict[str, object]
    baseline: Dict[str, float]
    greedy: Dict[str, object]
    milp: Dict[str, object]
    local_search_history: List[Dict[str, object]]
    final_validation: Dict[str, object]
    final_summary: Dict[str, object]
    candidate_solution_path: Path
    final_solution_path: Path
    artifacts: Dict[str, Path]


@dataclass(frozen=True)
class GreedySeedResult:
    optimizer: core.BonsaiOptimizer
    clusters: List[core.Cluster]
    summary: Dict[str, object]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def audit_procurement(
    catalog_path: Path,
    operations_path: Path,
    procurement_path: Path,
) -> Dict[str, object]:
    """Documenta por qué Operations se usa como demanda y Procurement como control.

    Los valores ERROR no se imputan para valorar la solución futura: los precios se
    recalculan desde el volumen consolidado y el tier. Aun así, la auditoría deja una
    evidencia reproducible de la decisión de fuente.
    """
    catalog = read_rows(catalog_path)
    operations = {row["codigo_producto"]: row for row in read_rows(operations_path)}
    procurement = read_rows(procurement_path)

    products_by_box: Dict[str, List[str]] = defaultdict(list)
    for row in catalog:
        products_by_box[row["caja_tipo_id"]].append(row["codigo_producto"])

    error_count = 0
    numeric_count = 0
    matches_operations = 0

    for row in procurement:
        box_id = row["caja_tipo_id"]
        codes = products_by_box.get(box_id, [])
        for plant in PLANTS:
            raw_price = str(row[f"costo_unitario_planta_{plant}"]).strip()
            if raw_price.upper() == "ERROR":
                error_count += 1
                continue

            numeric_count += 1
            observed = core.parse_float(raw_price)
            volume = sum(
                core.parse_float(operations[code][f"volumen_producto_planta_{plant}"])
                for code in codes
                if code in operations
            )
            expected = core.parse_float(row["costo_unitario_base"]) * core.tier_factor(volume)
            if math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-6):
                matches_operations += 1

    return {
        "procurement_error_cells": error_count,
        "procurement_numeric_price_cells": numeric_count,
        "prices_matching_operations_tiers": matches_operations,
        "matching_rate": matches_operations / numeric_count if numeric_count else None,
        "demand_source": "operaciones_planta",
        "procurement_role": "auditoria; los tiers de la solucion se recalculan",
    }


def products_from_prepared_data(data: bonsai_data.PreparedData) -> List[core.Product]:
    """Adapta la tabla validada de pandas al modelo de dominio del optimizador."""
    volume_columns = [f"volumen_producto_planta_{plant}" for plant in core.PLANTS]
    products: List[core.Product] = []

    for idx, row in enumerate(data.products.itertuples(index=False)):
        products.append(
            core.Product(
                idx=idx,
                code=row.codigo_producto,
                current_box_id=row.caja_tipo_id,
                orig_dims=(
                    row.caja_interior_largo,
                    row.caja_interior_ancho,
                    row.caja_interior_alto,
                ),
                weight_kg=row.peso_neto_caja,
                volumes=np.asarray(
                    [getattr(row, column) for column in volume_columns],
                    dtype=float,
                ),
            )
        )
    return products


def compute_greedy_seed(
    products: Sequence[core.Product],
    baseline: Dict[str, float],
    quiet: bool,
    *,
    thickness: float = THICKNESS,
    pallet_cost: float = PALLET_COST,
) -> GreedySeedResult:
    """Calcula la incumbente greedy/local sin usar archivos como interfaz."""
    optimizer = core.BonsaiOptimizer(products, thickness, pallet_costs=pallet_cost)
    clusters = optimizer.optimize_greedy(verbose=not quiet)
    clusters = optimizer.improve_local(clusters, verbose=not quiet)
    summary = core.summarize_solution(optimizer, clusters, baseline)
    return GreedySeedResult(optimizer=optimizer, clusters=list(clusters), summary=summary)


def greedy_seed_rows(result: GreedySeedResult) -> List[model.SubmissionRow]:
    """Materializa la asignación greedy con la misma precisión del CSV legacy."""
    assignment: Dict[int, core.Cluster] = {}
    for cluster in result.clusters:
        for idx in cluster.members:
            if idx in assignment:
                raise RuntimeError(f"Producto duplicado en clusters: {idx}")
            assignment[idx] = cluster
    if len(assignment) != len(result.optimizer.products):
        raise RuntimeError("No todos los productos quedaron asignados")

    rows: List[model.SubmissionRow] = []
    for product in result.optimizer.products:
        cluster = assignment[product.idx]
        l, w, h = cluster.dims
        rows.append(
            model.SubmissionRow(
                codigo_producto=product.code,
                caja_grosor_mm=float(f"{result.optimizer.thickness:.1f}"),
                caja_exterior_largo=float(f"{l + 2 * result.optimizer.thickness:.9f}"),
                caja_exterior_ancho=float(f"{w + 2 * result.optimizer.thickness:.9f}"),
                caja_exterior_alto=float(f"{h + 2 * result.optimizer.thickness:.9f}"),
            )
        )
    return rows


def build_greedy_seed(
    products: Sequence[core.Product],
    baseline: Dict[str, float],
    output_path: Path,
    quiet: bool,
    *,
    thickness: float = THICKNESS,
    pallet_cost: float = PALLET_COST,
) -> GreedySeedResult:
    """Backward-compatible artifact wrapper around the in-memory greedy result."""
    result = compute_greedy_seed(products, baseline, quiet=quiet, thickness=thickness, pallet_cost=pallet_cost)
    core.write_assignment_csv(output_path, result.optimizer, result.clusters)
    return result

def build_pair_candidates(
    products: Sequence[core.Product],
    candidates: Sequence[model.CandidateBox],
    quiet: bool,
    *,
    thickness: float = THICKNESS,
    pallet_cost: float = PALLET_COST,
) -> List[model.CandidateBox]:
    """Añade cajas definidas por pares porque suelen representar fronteras geométricas.

    No se generan grupos arbitrarios: las cajas de pares ya cubren la mayor parte de las
    intersecciones útiles y mantienen el catálogo candidato en un tamaño manejable.
    """
    optimizer = core.BonsaiOptimizer(products, thickness, pallet_costs=pallet_cost)
    pool = {candidate.key: candidate for candidate in candidates}
    feasible_pairs = 0

    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            cluster = optimizer.cluster_from_members((i, j))
            if cluster is None:
                continue
            feasible_pairs += 1
            candidate = model.CandidateBox(
                dims_internal=tuple(float(value) for value in cluster.dims),
                q=cluster.q,
                source="pair",
                thickness=thickness,
            )
            pool.setdefault(candidate.key, candidate)
        if not quiet and (i + 1) % 50 == 0:
            print(f"Pares procesados: {i + 1}/{len(products)}")

    if not quiet:
        print(f"Pares factibles: {feasible_pairs:,}; cajas candidatas únicas: {len(pool):,}")
    return list(pool.values())


class LocalSearch:
    """Refina una asignación sobre un catálogo fijo de cajas candidatas.

    Los subproblemas locales permiten modelar exactamente los saltos de tier sin resolver
    de una vez el MILP global, que contiene demasiadas combinaciones producto-caja.
    """

    def __init__(
        self,
        products: Sequence[core.Product],
        candidates: Sequence[model.CandidateBox],
        assignment: Dict[int, int],
        *,
        thickness: float = THICKNESS,
        pallet_cost: float = PALLET_COST,
        base_price: float = model.BASE_PRICE,
        ect: float = model.ECT,
        headspace_ratio: float = model.HEADSPACE_RATIO,
    ) -> None:
        self.products = list(products)
        self.candidates = list(candidates)
        self.assignment = dict(assignment)
        self.thickness = float(thickness)
        self.pallet_cost = float(pallet_cost)
        self.base_price = float(base_price)
        self.ect = float(ect)
        self.headspace_ratio = float(headspace_ratio)
        self.n_products = len(products)
        self.n_boxes = len(candidates)

        self.compatible = np.zeros((self.n_products, self.n_boxes), dtype=bool)
        self.compatible_by_product: List[List[int]] = [[] for _ in products]
        self.compatible_by_box: List[List[int]] = [[] for _ in candidates]

        capacities = sorted({candidate.q for candidate in candidates})
        self.freight_by_q = {
            q: np.array([model.freight_cost(product, q, self.pallet_cost) for product in products], dtype=float)
            for q in capacities
        }

        for box_idx, candidate in enumerate(candidates):
            for product_idx, product in enumerate(products):
                if model.is_compatible(product, candidate, thickness=self.thickness, ect=self.ect, headspace_ratio=self.headspace_ratio):
                    self.compatible[product_idx, box_idx] = True
                    self.compatible_by_product[product_idx].append(box_idx)
                    self.compatible_by_box[box_idx].append(product_idx)

        self._rebuild_state()

    def _packaging(self, volumes: np.ndarray) -> float:
        return core.packaging_cost(volumes, self.base_price)

    def _rebuild_state(self) -> None:
        self.volumes = np.zeros((self.n_boxes, N_PLANTS), dtype=float)
        self.members: List[Set[int]] = [set() for _ in range(self.n_boxes)]
        for product_idx, box_idx in self.assignment.items():
            self.volumes[box_idx] += self.products[product_idx].volumes
            self.members[box_idx].add(product_idx)
        self.packaging_by_box = np.array(
            [self._packaging(volumes) for volumes in self.volumes], dtype=float
        )

    def costs(self) -> Tuple[float, float, float]:
        packaging = float(self.packaging_by_box.sum())
        freight = float(
            sum(
                self.freight_by_q[self.candidates[box_idx].q][product_idx]
                for product_idx, box_idx in self.assignment.items()
            )
        )
        return packaging, freight, packaging + freight

    def used_boxes(self) -> List[int]:
        return [idx for idx, members in enumerate(self.members) if members]

    def _candidate_boxes_for_neighborhood(self, selected: Set[int]) -> List[int]:
        """Limita cada MILP a cajas plausibles para dedicar tiempo a decisiones útiles."""
        boxes = {self.assignment[i] for i in selected}

        for box_idx in self.used_boxes():
            if any(self.compatible[i, box_idx] for i in selected):
                boxes.add(box_idx)

        for product_idx in selected:
            compatible = self.compatible_by_product[product_idx]
            ranked = sorted(
                compatible,
                key=lambda box_idx: (
                    self.candidates[box_idx].q,
                    len(self.compatible_by_box[box_idx]),
                ),
                reverse=True,
            )
            boxes.update(ranked[:6])

        coverage = []
        for box_idx in range(self.n_boxes):
            count = sum(self.compatible[i, box_idx] for i in selected)
            if count >= 2:
                coverage.append((count, self.candidates[box_idx].q, box_idx))
        coverage.sort(reverse=True)
        boxes.update(item[-1] for item in coverage[:60])
        return sorted(boxes)

    def solve_neighborhood(
        self,
        selected: Iterable[int],
        time_limit: float,
    ) -> Dict[int, int] | None:
        selected = sorted(set(selected))
        selected_set = set(selected)
        boxes = self._candidate_boxes_for_neighborhood(selected_set)

        fixed_volumes = np.zeros((len(boxes), N_PLANTS), dtype=float)
        for local_box, box_idx in enumerate(boxes):
            fixed_products = self.members[box_idx] - selected_set
            if fixed_products:
                fixed_volumes[local_box] = np.sum(
                    [self.products[i].volumes for i in fixed_products], axis=0
                )

        edges: List[Tuple[int, int]] = []
        edge_costs: List[float] = []
        edges_by_product = {i: [] for i in selected}
        edges_by_box: List[List[Tuple[int, int]]] = [[] for _ in boxes]

        for product_idx in selected:
            for local_box, box_idx in enumerate(boxes):
                if not self.compatible[product_idx, box_idx]:
                    continue
                edge_idx = len(edges)
                edges.append((product_idx, local_box))
                edge_costs.append(
                    self.freight_by_q[self.candidates[box_idx].q][product_idx]
                )
                edges_by_product[product_idx].append(edge_idx)
                edges_by_box[local_box].append((edge_idx, product_idx))

        if any(not edges_by_product[i] for i in selected):
            return None

        n_edges = len(edges)
        v_start = n_edges
        y_start = v_start + len(boxes) * N_PLANTS * N_TIERS
        n_variables = y_start + len(boxes) * N_PLANTS * N_TIERS

        objective = np.zeros(n_variables, dtype=float)
        objective[:n_edges] = edge_costs
        for local_box in range(len(boxes)):
            for plant_idx in range(N_PLANTS):
                for tier_idx, factor in enumerate(model.TIER_FACTORS):
                    objective[
                        v_start + (local_box * N_PLANTS + plant_idx) * N_TIERS + tier_idx
                    ] = self.base_price * factor

        integrality = np.zeros(n_variables, dtype=int)
        integrality[:n_edges] = 1
        integrality[y_start:] = 1

        lower_bounds = np.zeros(n_variables, dtype=float)
        upper_bounds = np.full(n_variables, np.inf, dtype=float)
        upper_bounds[:n_edges] = 1.0
        upper_bounds[y_start:] = 1.0

        row_idx: List[int] = []
        col_idx: List[int] = []
        values: List[float] = []
        constraint_lower: List[float] = []
        constraint_upper: List[float] = []
        row = 0

        def add(column: int, value: float) -> None:
            row_idx.append(row)
            col_idx.append(column)
            values.append(value)

        for product_idx in selected:
            for edge_idx in edges_by_product[product_idx]:
                add(edge_idx, 1.0)
            constraint_lower.append(1.0)
            constraint_upper.append(1.0)
            row += 1

        selected_totals = np.sum([self.products[i].volumes for i in selected], axis=0)
        for local_box in range(len(boxes)):
            for plant_idx in range(N_PLANTS):
                for edge_idx, product_idx in edges_by_box[local_box]:
                    volume = float(self.products[product_idx].volumes[plant_idx])
                    if volume:
                        add(edge_idx, volume)
                for tier_idx in range(N_TIERS):
                    add(
                        v_start
                        + (local_box * N_PLANTS + plant_idx) * N_TIERS
                        + tier_idx,
                        -1.0,
                    )
                fixed = float(fixed_volumes[local_box, plant_idx])
                constraint_lower.append(-fixed)
                constraint_upper.append(-fixed)
                row += 1

                maximum_volume = fixed + float(selected_totals[plant_idx])
                tier_uppers = (
                    *model.TIER_UPPERS_FIXED,
                    max(500_000.0, maximum_volume),
                )
                for tier_idx in range(N_TIERS):
                    volume_idx = (
                        v_start
                        + (local_box * N_PLANTS + plant_idx) * N_TIERS
                        + tier_idx
                    )
                    binary_idx = (
                        y_start
                        + (local_box * N_PLANTS + plant_idx) * N_TIERS
                        + tier_idx
                    )
                    add(volume_idx, 1.0)
                    add(binary_idx, -model.TIER_LOWERS[tier_idx])
                    constraint_lower.append(0.0)
                    constraint_upper.append(np.inf)
                    row += 1

                    add(volume_idx, 1.0)
                    add(binary_idx, -tier_uppers[tier_idx])
                    constraint_lower.append(-np.inf)
                    constraint_upper.append(0.0)
                    row += 1

                for tier_idx in range(N_TIERS):
                    add(
                        y_start
                        + (local_box * N_PLANTS + plant_idx) * N_TIERS
                        + tier_idx,
                        1.0,
                    )
                constraint_lower.append(1.0 if fixed > 0 else -np.inf)
                constraint_upper.append(1.0)
                row += 1

        matrix = coo_matrix(
            (values, (row_idx, col_idx)), shape=(row, n_variables)
        ).tocsr()
        result = milp(
            objective,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=LinearConstraint(
                matrix,
                np.asarray(constraint_lower),
                np.asarray(constraint_upper),
            ),
            options={
                "time_limit": float(time_limit),
                "mip_rel_gap": 1e-6,
                "presolve": True,
                "disp": False,
            },
        )
        if result.x is None:
            return None

        new_assignment: Dict[int, int] = {}
        for edge_idx, value in enumerate(result.x[:n_edges]):
            if value > 0.5:
                product_idx, local_box = edges[edge_idx]
                new_assignment[product_idx] = boxes[local_box]
        return new_assignment if len(new_assignment) == len(selected) else None

    def try_assignment(self, changes: Dict[int, int]) -> float:
        before = self.costs()[2]
        previous = {i: self.assignment[i] for i in changes}
        for product_idx, box_idx in changes.items():
            self.assignment[product_idx] = box_idx
        self._rebuild_state()
        after = self.costs()[2]

        if after < before - 1e-6:
            return before - after

        for product_idx, box_idx in previous.items():
            self.assignment[product_idx] = box_idx
        self._rebuild_state()
        return 0.0

    def improve(self, rounds: int, time_limit: float, quiet: bool) -> List[Dict[str, object]]:
        history: List[Dict[str, object]] = []
        random.seed(20260727)

        for round_idx in range(rounds):
            neighborhoods: List[Set[int]] = []
            thresholds = (20_000, 50_000, 100_000, 500_000)

            # Los grupos próximos a un tier son los que más pueden beneficiarse de mover
            # varios productos a la vez.
            near_tier = []
            for box_idx in self.used_boxes():
                for plant_idx, volume in enumerate(self.volumes[box_idx]):
                    next_threshold = next((t for t in thresholds if volume < t), None)
                    if next_threshold is not None:
                        near_tier.append(
                            ((next_threshold - volume) / next_threshold, box_idx, plant_idx)
                        )

            for _, target_box, plant_idx in sorted(near_tier)[:12]:
                selected = set(self.members[target_box])
                external = []
                for product_idx in self.compatible_by_box[target_box]:
                    if product_idx in selected:
                        continue
                    contribution = self.products[product_idx].volumes[plant_idx]
                    if contribution <= 0:
                        continue
                    current_box = self.assignment[product_idx]
                    freight_penalty = (
                        self.freight_by_q[self.candidates[target_box].q][product_idx]
                        - self.freight_by_q[self.candidates[current_box].q][product_idx]
                    )
                    external.append(
                        (freight_penalty / max(contribution, 1.0), -contribution, product_idx)
                    )
                for *_, product_idx in sorted(external)[:24]:
                    selected.add(product_idx)
                if 2 <= len(selected) <= 55:
                    neighborhoods.append(selected)

            used = self.used_boxes()
            small_boxes = [box_idx for box_idx in used if len(self.members[box_idx]) <= 15]
            for _ in range(10):
                random.shuffle(small_boxes)
                selected: Set[int] = set()
                for box_idx in small_boxes:
                    if len(selected) + len(self.members[box_idx]) <= 50:
                        selected.update(self.members[box_idx])
                    if len(selected) >= 25:
                        break
                if len(selected) >= 2:
                    neighborhoods.append(selected)

            unique = []
            seen = set()
            for selected in neighborhoods:
                key = tuple(sorted(selected))
                if key not in seen:
                    seen.add(key)
                    unique.append(selected)

            improved = False
            for selected in unique:
                candidate_assignment = self.solve_neighborhood(selected, time_limit)
                if candidate_assignment is None:
                    continue
                saving = self.try_assignment(candidate_assignment)
                if saving > 0:
                    improved = True
                    record = {
                        "round": round_idx + 1,
                        "products": len(selected),
                        "saving": saving,
                        "total": self.costs()[2],
                        "types": len(self.used_boxes()),
                    }
                    history.append(record)
                    if not quiet:
                        print(f"Mejora LNS: USD {saving:,.2f}; total USD {record['total']:,.2f}")

            if not improved:
                break
        return history

    def improve_single_target(self, max_targets: int, time_limit: float, quiet: bool) -> Dict[str, object] | None:
        """Busca saltos de tier al trasladar un subconjunto hacia una única caja destino."""
        used = set(self.used_boxes())

        def target_score(box_idx: int) -> Tuple[object, ...]:
            external = [i for i in self.compatible_by_box[box_idx] if self.assignment[i] != box_idx]
            no_penalty = [
                i
                for i in external
                if self.freight_by_q[self.candidates[box_idx].q][i]
                <= self.freight_by_q[self.candidates[self.assignment[i]].q][i] + 1e-9
            ]
            volume = sum(float(self.products[i].volumes.sum()) for i in no_penalty)
            return (
                box_idx in used,
                len(no_penalty),
                volume,
                len({self.assignment[i] for i in external}),
                self.candidates[box_idx].q,
            )

        ranked = sorted(range(self.n_boxes), key=target_score, reverse=True)
        targets = list(dict.fromkeys([*used, *ranked[:max_targets]]))
        best_changes = None
        best_saving = 0.0

        for target_box in targets:
            external = [
                i for i in self.compatible_by_box[target_box] if self.assignment[i] != target_box
            ]
            if not external:
                continue

            # Limitar candidatos evita que una sola caja muy genérica domine el tiempo
            # de ejecución sin aportar una decisión más informativa.
            if len(external) > 80:
                external = sorted(
                    external,
                    key=lambda i: (
                        self.freight_by_q[self.candidates[target_box].q][i]
                        - self.freight_by_q[self.candidates[self.assignment[i]].q][i],
                        -float(self.products[i].volumes.sum()),
                    ),
                )[:80]

            # Este MILP decide qué subset conviene mover; el resto permanece intacto.
            source_boxes = sorted({self.assignment[i] for i in external})
            boxes = [target_box] + [box for box in source_boxes if box != target_box]
            local_index = {box: idx for idx, box in enumerate(boxes)}
            n_binary = len(external)
            v_start = n_binary
            y_start = v_start + len(boxes) * N_PLANTS * N_TIERS
            n_variables = y_start + len(boxes) * N_PLANTS * N_TIERS

            objective = np.zeros(n_variables)
            for idx, product_idx in enumerate(external):
                objective[idx] = (
                    self.freight_by_q[self.candidates[target_box].q][product_idx]
                    - self.freight_by_q[
                        self.candidates[self.assignment[product_idx]].q
                    ][product_idx]
                )
            for local_box in range(len(boxes)):
                for plant_idx in range(N_PLANTS):
                    for tier_idx, factor in enumerate(model.TIER_FACTORS):
                        objective[
                            v_start
                            + (local_box * N_PLANTS + plant_idx) * N_TIERS
                            + tier_idx
                        ] = self.base_price * factor

            integrality = np.zeros(n_variables, dtype=int)
            integrality[:n_binary] = 1
            integrality[y_start:] = 1
            lower = np.zeros(n_variables)
            upper = np.full(n_variables, np.inf)
            upper[:n_binary] = 1.0
            upper[y_start:] = 1.0

            row_idx: List[int] = []
            col_idx: List[int] = []
            values: List[float] = []
            lowers: List[float] = []
            uppers: List[float] = []
            row = 0

            def add(column: int, value: float) -> None:
                row_idx.append(row)
                col_idx.append(column)
                values.append(value)

            for box in boxes:
                local_box = local_index[box]
                for plant_idx in range(N_PLANTS):
                    base_volume = float(self.volumes[box, plant_idx])
                    if box == target_box:
                        for idx, product_idx in enumerate(external):
                            volume = float(self.products[product_idx].volumes[plant_idx])
                            if volume:
                                add(idx, volume)
                    else:
                        for idx, product_idx in enumerate(external):
                            if self.assignment[product_idx] == box:
                                volume = float(self.products[product_idx].volumes[plant_idx])
                                if volume:
                                    add(idx, -volume)

                    for tier_idx in range(N_TIERS):
                        add(
                            v_start
                            + (local_box * N_PLANTS + plant_idx) * N_TIERS
                            + tier_idx,
                            -1.0,
                        )
                    lowers.append(-base_volume)
                    uppers.append(-base_volume)
                    row += 1

                    max_volume = base_volume + sum(
                        float(self.products[i].volumes[plant_idx]) for i in external
                    )
                    tier_uppers = (
                        *model.TIER_UPPERS_FIXED,
                        max(500_000.0, max_volume),
                    )
                    for tier_idx in range(N_TIERS):
                        volume_idx = (
                            v_start
                            + (local_box * N_PLANTS + plant_idx) * N_TIERS
                            + tier_idx
                        )
                        binary_idx = (
                            y_start
                            + (local_box * N_PLANTS + plant_idx) * N_TIERS
                            + tier_idx
                        )
                        add(volume_idx, 1.0)
                        add(binary_idx, -model.TIER_LOWERS[tier_idx])
                        lowers.append(0.0)
                        uppers.append(np.inf)
                        row += 1

                        add(volume_idx, 1.0)
                        add(binary_idx, -tier_uppers[tier_idx])
                        lowers.append(-np.inf)
                        uppers.append(0.0)
                        row += 1

                    for tier_idx in range(N_TIERS):
                        add(
                            y_start
                            + (local_box * N_PLANTS + plant_idx) * N_TIERS
                            + tier_idx,
                            1.0,
                        )
                    lowers.append(-np.inf)
                    uppers.append(1.0)
                    row += 1

            matrix = coo_matrix(
                (values, (row_idx, col_idx)), shape=(row, n_variables)
            ).tocsr()
            result = milp(
                objective,
                integrality=integrality,
                bounds=Bounds(lower, upper),
                constraints=LinearConstraint(
                    matrix, np.asarray(lowers), np.asarray(uppers)
                ),
                options={
                    "time_limit": float(time_limit),
                    "mip_rel_gap": 1e-7,
                    "presolve": True,
                    "disp": False,
                },
            )
            if result.x is None:
                continue

            moved = [
                product_idx
                for idx, product_idx in enumerate(external)
                if result.x[idx] > 0.5
            ]
            if not moved:
                continue

            changes = {product_idx: target_box for product_idx in moved}
            previous = {i: self.assignment[i] for i in changes}
            before = self.costs()[2]
            for product_idx, box_idx in changes.items():
                self.assignment[product_idx] = box_idx
            self._rebuild_state()
            saving = before - self.costs()[2]
            for product_idx, box_idx in previous.items():
                self.assignment[product_idx] = box_idx
            self._rebuild_state()

            if saving > best_saving + 1e-6:
                best_saving = saving
                best_changes = changes

        if not best_changes:
            return None

        self.try_assignment(best_changes)
        result = {
            "moved_products": len(best_changes),
            "saving": best_saving,
            "total": self.costs()[2],
            "types": len(self.used_boxes()),
        }
        if not quiet:
            print(f"Mejora por subset: USD {best_saving:,.2f}")
        return result


def write_raw_submission(
    path: Path,
    products: Sequence[core.Product],
    candidates: Sequence[model.CandidateBox],
    assignment: Dict[int, int],
) -> List[model.SubmissionRow]:
    """Materializa la solución candidata y conserva el CSV como artefacto."""
    rows = model.submission_rows(products, candidates, assignment)
    model.write_submission_rows(path, rows)
    return rows


def round_solution_rows(
    raw_rows: Sequence[model.SubmissionRow],
    catalog_path: Path,
    specs_path: Path,
    *,
    thickness: float = THICKNESS,
    ect: float = model.ECT,
    headspace_ratio: float = model.HEADSPACE_RATIO,
) -> List[model.SubmissionRow]:
    """Redondea a 0,01 mm sin perder factibilidad, operando en memoria."""
    catalog = read_rows(catalog_path)
    catalog_by_code = {row["codigo_producto"]: row for row in catalog}
    specs = {row["caja_tipo_id"]: row for row in read_rows(specs_path)}

    groups: Dict[Tuple[float, float, float], List[str]] = defaultdict(list)
    for row in raw_rows:
        key = (
            float(row.caja_exterior_largo),
            float(row.caja_exterior_ancho),
            float(row.caja_exterior_alto),
        )
        groups[key].append(row.codigo_producto)

    def capacity(ext: Sequence[float]) -> int:
        return (
            math.floor((core.PALLET_ANCHO + 1e-9) / ext[0])
            * math.floor((core.PALLET_LARGO + 1e-9) / ext[1])
            * math.floor((core.PALLET_ALTO + 1e-9) / ext[2])
        )

    def feasible(ext: Sequence[float], codes: Sequence[str]) -> bool:
        internal = [value - 2.0 * thickness for value in ext]
        if min(internal) <= 0:
            return False
        layers = math.floor((core.PALLET_ALTO + 1e-9) / ext[2])
        load_max = ect * (2.0 * (ext[0] + ext[1]) / 1000.0) / core.G

        for code in codes:
            item = catalog_by_code[code]
            spec = specs[item["caja_tipo_id"]]
            original = [
                core.parse_float(spec["caja_interior_largo"]),
                core.parse_float(spec["caja_interior_ancho"]),
                core.parse_float(spec["caja_interior_alto"]),
            ]
            for new, old in zip(internal, original):
                if new < 0.90 * old - 1e-5 or new > 1.10 * old + 1e-5:
                    return False
                if new - old > min(headspace_ratio * new, 40.0) + 1e-5:
                    return False
            if math.prod(internal) + 1e-5 < math.prod(original):
                return False
            required = max(0, layers - 1) * core.parse_float(item["peso_neto_caja"])
            if load_max + 1e-5 < required:
                return False
        return True

    rounded: Dict[Tuple[float, float, float], Tuple[float, float, float]] = {}
    for raw_ext, codes in groups.items():
        raw_capacity = capacity(raw_ext)
        best = None
        for radius in (3, 10, 30, 100):
            values = []
            for value in raw_ext:
                center = round(value * 100)
                values.append(
                    sorted({(center + delta) / 100 for delta in range(-radius, radius + 1)})
                )
            for ext in itertools.product(*values):
                if capacity(ext) < raw_capacity or not feasible(ext, codes):
                    continue
                distance = sum((ext[axis] - raw_ext[axis]) ** 2 for axis in range(3))
                score = (distance, -capacity(ext), math.prod(ext))
                if best is None or score < best[0]:
                    best = (score, ext)
            if best is not None:
                break
        if best is None:
            raise RuntimeError(f"No se pudo redondear de forma factible la caja {raw_ext}")
        rounded[raw_ext] = best[1]

    result: List[model.SubmissionRow] = []
    for row in raw_rows:
        raw_key = (
            float(row.caja_exterior_largo),
            float(row.caja_exterior_ancho),
            float(row.caja_exterior_alto),
        )
        ext = rounded[raw_key]
        result.append(
            model.SubmissionRow(
                codigo_producto=row.codigo_producto,
                caja_grosor_mm=thickness,
                caja_exterior_largo=float(f"{ext[0]:.2f}"),
                caja_exterior_ancho=float(f"{ext[1]:.2f}"),
                caja_exterior_alto=float(f"{ext[2]:.2f}"),
            )
        )
    return result


def _read_submission_rows(path: Path) -> List[model.SubmissionRow]:
    return [
        model.SubmissionRow(
            codigo_producto=row["codigo_producto"],
            caja_grosor_mm=core.parse_float(row["caja_grosor_mm"]),
            caja_exterior_largo=core.parse_float(row["caja_exterior_largo"]),
            caja_exterior_ancho=core.parse_float(row["caja_exterior_ancho"]),
            caja_exterior_alto=core.parse_float(row["caja_exterior_alto"]),
        )
        for row in read_rows(path)
    ]


def _write_rounded_submission_rows(path: Path, rows: Sequence[model.SubmissionRow]) -> None:
    fields = [
        "codigo_producto",
        "caja_grosor_mm",
        "caja_exterior_largo",
        "caja_exterior_ancho",
        "caja_exterior_alto",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "codigo_producto": row.codigo_producto,
                    "caja_grosor_mm": f"{row.caja_grosor_mm:.2f}",
                    "caja_exterior_largo": f"{row.caja_exterior_largo:.2f}",
                    "caja_exterior_ancho": f"{row.caja_exterior_ancho:.2f}",
                    "caja_exterior_alto": f"{row.caja_exterior_alto:.2f}",
                }
            )


def round_submission_feasibly(
    raw_path: Path,
    output_path: Path,
    catalog_path: Path,
    specs_path: Path,
) -> None:
    """Legacy file wrapper around :func:`round_solution_rows`."""
    rounded = round_solution_rows(_read_submission_rows(raw_path), catalog_path, specs_path)
    _write_rounded_submission_rows(output_path, rounded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--catalog", default="catalogo_productos.csv")
    parser.add_argument("--specs", default="especificaciones_cajas.csv")
    parser.add_argument("--operations", default="operaciones_planta.csv")
    parser.add_argument("--procurement", default="procurement_cajas.csv")
    parser.add_argument("--milp-time", type=float, default=30.0)
    parser.add_argument("--lns-time", type=float, default=0.2)
    parser.add_argument("--lns-rounds", type=int, default=2)
    parser.add_argument("--target-count", type=int, default=40)
    parser.add_argument("--target-time", type=float, default=0.05)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Omite LNS; sirve para comprobar el pipeline en pocos minutos.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> OptimizationConfig:
    return OptimizationConfig(
        milp_time=args.milp_time,
        lns_time=args.lns_time,
        lns_rounds=args.lns_rounds,
        target_count=args.target_count,
        target_time=args.target_time,
        quiet=args.quiet,
        fast=args.fast,
    )


def optimize_portfolio(
    catalog_path: Path,
    specs_path: Path,
    operations_path: Path,
    procurement_path: Path,
    output_dir: Path,
    config: OptimizationConfig | None = None,
    problem: OptimizationProblem | None = None,
) -> OptimizationResult:
    """Ejecuta el pipeline determinístico completo sin depender del CLI."""
    problem = problem or default_bonsai_packaging_problem()
    if config is None:
        solver = problem.solver_parameters
        config = OptimizationConfig(
            milp_time=solver["milp_time"],
            mip_gap=solver["mip_gap"],
            extra_candidates=solver["extra_candidates"],
            lns_time=solver["lns_time"],
            lns_rounds=solver["lns_rounds"],
            target_count=solver["target_count"],
            target_time=solver["target_time"],
        )
    thickness = float(problem.values["thickness"])
    pallet_cost = float(problem.values["pallet_cost"])
    problem_cfg = core.THICKNESS_CONFIG[thickness]
    base_price = float(problem_cfg["base_price"])
    ect = float(problem_cfg["ect"])
    headspace_ratio = float(problem_cfg["headspace"])
    catalog_path = Path(catalog_path)
    specs_path = Path(specs_path)
    operations_path = Path(operations_path)
    procurement_path = Path(procurement_path)
    output_dir = Path(output_dir)

    for path in (catalog_path, specs_path, operations_path, procurement_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = bonsai_data.load_inputs(
        catalog_path, specs_path, operations_path, procurement_path
    )
    audit, _audit_detail = bonsai_data.audit_procurement(prepared)
    audit_path = output_dir / "auditoria_datos.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    products = products_from_prepared_data(prepared)
    baseline = bonsai_data.baseline_from_operations(prepared.operations)

    seed_path = output_dir / "01_seed_greedy.csv"
    greedy_result = build_greedy_seed(products, baseline, seed_path, quiet=config.quiet, thickness=thickness, pallet_cost=pallet_cost)
    if isinstance(greedy_result, GreedySeedResult):
        greedy_summary = greedy_result.summary
        seed_candidates, seed_keys = model.seed_candidates_from_rows(greedy_seed_rows(greedy_result), thickness=thickness)
    else:
        # Transitional compatibility for legacy/custom wrappers returning only a summary.
        greedy_summary = greedy_result
        seed_candidates, seed_keys = model.read_seed_candidates(seed_path, thickness=thickness)

    compact_candidates = model.build_candidate_pool(
        products, seed_candidates, seed_keys, extra_candidates=config.extra_candidates,
        thickness=thickness, pallet_cost=pallet_cost, ect=ect,
        headspace_ratio=headspace_ratio, base_price=base_price,
    )
    assignment, milp_result, _, _, _ = model.solve_assignment_milp(
        products, compact_candidates, time_limit=config.milp_time, mip_gap=config.mip_gap,
        pallet_cost=pallet_cost, base_price=base_price, thickness=thickness,
        ect=ect, headspace_ratio=headspace_ratio,
    )
    milp_costs = model.exact_costs(products, compact_candidates, assignment, base_price=base_price, pallet_cost=pallet_cost)
    milp_raw_path = output_dir / "02_submission_milp_raw.csv"
    model.write_submission(milp_raw_path, products, compact_candidates, assignment)

    milp_summary: Dict[str, object] = {
        "status": int(milp_result.status),
        "message": str(milp_result.message),
        "packaging": milp_costs[0],
        "freight": milp_costs[1],
        "total": milp_costs[2],
        "types": len(set(assignment.values())),
    }

    history: List[Dict[str, object]] = []
    final_candidates = compact_candidates
    final_assignment = assignment
    if not config.fast:
        final_candidates = build_pair_candidates(products, compact_candidates, quiet=config.quiet, thickness=thickness, pallet_cost=pallet_cost)
        key_to_box = {candidate.key: idx for idx, candidate in enumerate(final_candidates)}
        compact_keys = {idx: candidate.key for idx, candidate in enumerate(compact_candidates)}
        final_assignment = {
            product_idx: key_to_box[compact_keys[box_idx]]
            for product_idx, box_idx in assignment.items()
        }
        search = LocalSearch(products, final_candidates, final_assignment, thickness=thickness, pallet_cost=pallet_cost, base_price=base_price, ect=ect, headspace_ratio=headspace_ratio)
        history.extend(
            search.improve(rounds=config.lns_rounds, time_limit=config.lns_time, quiet=config.quiet)
        )
        subset_result = search.improve_single_target(
            max_targets=config.target_count, time_limit=config.target_time, quiet=config.quiet
        )
        if subset_result:
            history.append({"stage": "target_subset", **subset_result})
        final_assignment = search.assignment

    raw_path = output_dir / "submission_optimizada_raw.csv"
    final_path = output_dir / "submission_optimizada.csv"
    raw_rows = write_raw_submission(raw_path, products, final_candidates, final_assignment)
    if raw_rows is None:
        # Transitional compatibility for legacy/custom file-only wrappers.
        round_submission_feasibly(raw_path, final_path, catalog_path, specs_path)
    else:
        rounded_rows = round_solution_rows(raw_rows, catalog_path, specs_path, thickness=thickness, ect=ect, headspace_ratio=headspace_ratio)
        _write_rounded_submission_rows(final_path, rounded_rows)

    validation = validate(
        final_path, catalog_path, specs_path, operations_path, mode="faq", pallet_cost=pallet_cost
    )
    validation_path = output_dir / "validacion_submission.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary: Dict[str, object] = {
        "data_audit": audit,
        "baseline": baseline,
        "greedy": greedy_summary,
        "milp": milp_summary,
        "local_search_history": history,
        "final": validation,
    }
    artifacts: Dict[str, Path] = {
        "data_audit": audit_path,
        "greedy_seed": seed_path,
        "milp_raw": milp_raw_path,
        "candidate_raw": raw_path,
        "final_submission": final_path,
        "validation": validation_path,
    }
    success = bool(validation.get("valid"))
    if success:
        summary_path = output_dir / "resumen_optimizacion.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        artifacts["summary"] = summary_path

    return OptimizationResult(
        success=success,
        config=config,
        problem=problem,
        data_audit=audit,
        baseline=baseline,
        greedy=greedy_summary,
        milp=milp_summary,
        local_search_history=history,
        final_validation=validation,
        final_summary=summary,
        candidate_solution_path=raw_path,
        final_solution_path=final_path,
        artifacts=artifacts,
    )

def main() -> None:
    args = build_parser().parse_args()
    result = optimize_portfolio(
        catalog_path=args.data_dir / args.catalog,
        specs_path=args.data_dir / args.specs,
        operations_path=args.data_dir / args.operations,
        procurement_path=args.data_dir / args.procurement,
        output_dir=args.output_dir,
        config=config_from_args(args),
    )
    if not result.success:
        raise RuntimeError("La submission final no superó la validación independiente")
    print(json.dumps(result.final_validation, indent=2, ensure_ascii=False))
    print(f"\nCSV final: {result.final_solution_path.resolve()}")


if __name__ == "__main__":
    main()
