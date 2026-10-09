#!/usr/bin/env python3
"""
Optimización heurística del portafolio de cajas de Bonsai Corp.

Enfoque:
1) Usa Operations como fuente de demanda por SKU/planta.
2) Prueba por separado cada grosor uniforme permitido: 3.0, 4.5 y 5.0 mm.
3) Para cada conjunto de productos, busca la caja común con mayor capacidad de pallet
   entre los puntos de quiebre discretos del pallet.
4) Consolida clusters mediante greedy agglomerative merging, aceptando exclusivamente
   merges con ahorro positivo en packaging + flete.
5) Valida fit ±10%, volumen interno mínimo, headspace por eje, orientación fija,
   pallet 1200x800x1800 y resistencia ECT.

No requiere pandas. Dependencias: Python 3 y numpy.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PLANTS = ("buenos_aires", "curitiba", "santiago", "monterrey", "bakersfield")
PALLET_LARGO = 1200.0
PALLET_ANCHO = 800.0
PALLET_ALTO = 1800.0
G = 9.81
EPS = 1e-5

THICKNESS_CONFIG = {
    3.0: {"ect": 1000.0, "base_price": 0.60, "headspace": 0.06},
    4.5: {"ect": 1400.0, "base_price": 0.65, "headspace": 0.08},
    5.0: {"ect": 1650.0, "base_price": 0.70, "headspace": 0.10},
}


def parse_float(value: str) -> float:
    if value is None:
        raise ValueError("Valor numérico ausente")
    text = str(value).strip().lower().replace("mm", "").replace(" ", "").replace(",", ".")
    if text in {"", "nan", "none", "error"}:
        raise ValueError(f"Valor no numérico: {value!r}")
    return float(text)


def parse_int(value: str) -> int:
    return int(round(parse_float(value)))


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


def packaging_cost(volumes_by_plant: np.ndarray, base_price: float) -> float:
    total = 0.0
    for volume in volumes_by_plant:
        if volume > 0:
            total += float(volume) * base_price * tier_factor(float(volume))
    return total


def ceil_div_volume(volume: float, capacity: int) -> int:
    if volume <= 0:
        return 0
    return int(math.ceil((float(volume) - 1e-12) / capacity))


@dataclass(frozen=True)
class Product:
    idx: int
    code: str
    current_box_id: str
    orig_dims: Tuple[float, float, float]  # L, W, H internas
    weight_kg: float
    volumes: np.ndarray  # por planta

    @property
    def orig_volume(self) -> float:
        l, w, h = self.orig_dims
        return l * w * h


@dataclass
class Cluster:
    id: int
    members: Tuple[int, ...]
    lo: np.ndarray
    hi: np.ndarray
    max_product_volume: float
    max_weight: float
    volumes_by_plant: np.ndarray
    freight_by_q: np.ndarray
    q: int
    dims: Tuple[float, float, float]
    cost_packaging: float
    cost_freight: float
    total_cost: float
    version: int = 0
    active: bool = True


@dataclass(frozen=True)
class BoxSolution:
    q: int
    dims: Tuple[float, float, float]
    counts: Tuple[int, int, int]  # largo sobre 800, ancho sobre 1200, capas
    utilization: float


class BonsaiOptimizer:
    def __init__(
        self,
        products: Sequence[Product],
        thickness: float,
        pallet_costs: Sequence[float] | float = 150.0,
    ) -> None:
        if thickness not in THICKNESS_CONFIG:
            raise ValueError(f"Grosor no permitido: {thickness}")
        self.products = list(products)
        self.thickness = float(thickness)
        cfg = THICKNESS_CONFIG[self.thickness]
        self.ect = cfg["ect"]
        self.base_price = cfg["base_price"]
        self.headspace_ratio = cfg["headspace"]

        if isinstance(pallet_costs, (int, float)):
            self.pallet_costs = np.full(len(PLANTS), float(pallet_costs), dtype=float)
        else:
            arr = np.asarray(pallet_costs, dtype=float)
            if arr.shape != (len(PLANTS),):
                raise ValueError("pallet_costs debe tener un valor por planta")
            self.pallet_costs = arr

        self.product_lo = np.zeros((len(products), 3), dtype=float)
        self.product_hi = np.zeros((len(products), 3), dtype=float)
        for p in products:
            orig = np.asarray(p.orig_dims, dtype=float)
            self.product_lo[p.idx] = 0.90 * orig
            # Restricciones simultáneas: +10%, headspace porcentual y tope 40 mm.
            self.product_hi[p.idx] = np.minimum.reduce(
                [
                    1.10 * orig,
                    orig / (1.0 - self.headspace_ratio),
                    orig + 40.0,
                ]
            )

        # Cota global de capacidades posibles.
        min_lo = self.product_lo.min(axis=0)
        max_n_l = int(PALLET_ANCHO // (min_lo[0] + 2 * self.thickness))
        max_n_w = int(PALLET_LARGO // (min_lo[1] + 2 * self.thickness))
        max_n_h = int(PALLET_ALTO // (min_lo[2] + 2 * self.thickness))
        self.q_max = max_n_l * max_n_w * max_n_h

        triples = []
        for n_l in range(1, max_n_l + 1):
            for n_w in range(1, max_n_w + 1):
                for n_h in range(1, max_n_h + 1):
                    triples.append((n_l * n_w * n_h, n_l, n_w, n_h))
        # Mayor capacidad primero; desempate con más cajas por capa y menos capas.
        triples.sort(key=lambda x: (-x[0], -(x[1] * x[2]), x[3]))
        self.count_triples = triples

        # Flete por producto para cada q, luego los clusters solo suman vectores.
        self.product_freight_by_q = np.zeros((len(products), self.q_max + 1), dtype=float)
        for p in products:
            for q in range(1, self.q_max + 1):
                pallets = np.array([ceil_div_volume(v, q) for v in p.volumes], dtype=float)
                self.product_freight_by_q[p.idx, q] = float(np.dot(pallets, self.pallet_costs))

        self._next_cluster_id = 0

    def _new_cluster_id(self) -> int:
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        return cid

    def _exact_count_interval(
        self,
        group_lo: float,
        group_hi: float,
        pallet_dim: float,
        count: int,
    ) -> Optional[Tuple[float, float]]:
        # d_int + 2t debe cumplir pallet/(n+1) < d_ext <= pallet/n.
        lo = max(group_lo, pallet_dim / (count + 1) - 2 * self.thickness + EPS)
        hi = min(group_hi, pallet_dim / count - 2 * self.thickness - EPS)
        if lo > hi + 1e-9:
            return None
        return lo, hi

    def best_box(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        max_product_volume: float,
        max_weight: float,
    ) -> Optional[BoxSolution]:
        if np.any(lo > hi + 1e-9):
            return None

        for q_nominal, n_l, n_w, n_h in self.count_triples:
            int_l = self._exact_count_interval(lo[0], hi[0], PALLET_ANCHO, n_l)
            if int_l is None:
                continue
            int_w = self._exact_count_interval(lo[1], hi[1], PALLET_LARGO, n_w)
            if int_w is None:
                continue
            int_h = self._exact_count_interval(lo[2], hi[2], PALLET_ALTO, n_h)
            if int_h is None:
                continue

            lo_l, hi_l = int_l
            lo_w, hi_w = int_w
            lo_h, hi_h = int_h

            # Elegimos L y W altos dentro del mismo patrón para maximizar perímetro/ECT;
            # H es el mínimo que permite volumen suficiente.
            d_l = hi_l
            d_w = hi_w
            required_h = max_product_volume / (d_l * d_w)
            # Margen mínimo para que el redondeo del CSV no deje el volumen por debajo.
            d_h = max(lo_h, required_h + 1e-7)
            if d_h > hi_h + 1e-8:
                continue

            ext_l = d_l + 2 * self.thickness
            ext_w = d_w + 2 * self.thickness
            ext_h = d_h + 2 * self.thickness

            actual_n_l = int(math.floor((PALLET_ANCHO + 1e-9) / ext_l))
            actual_n_w = int(math.floor((PALLET_LARGO + 1e-9) / ext_w))
            actual_n_h = int(math.floor((PALLET_ALTO + 1e-9) / ext_h))
            actual_q = actual_n_l * actual_n_w * actual_n_h
            if (actual_n_l, actual_n_w, actual_n_h) != (n_l, n_w, n_h):
                continue
            if actual_q != q_nominal:
                continue

            load_max_kg = self.ect * (2.0 * (ext_l + ext_w) / 1000.0) / G
            required_load_kg = max(0, actual_n_h - 1) * max_weight
            if load_max_kg + 1e-9 < required_load_kg:
                continue

            box_ext_volume = ext_l * ext_w * ext_h
            pallet_util = actual_q * box_ext_volume / (PALLET_ANCHO * PALLET_LARGO * PALLET_ALTO)
            return BoxSolution(
                q=actual_q,
                dims=(d_l, d_w, d_h),
                counts=(actual_n_l, actual_n_w, actual_n_h),
                utilization=pallet_util,
            )
        return None

    def _build_cluster_from_stats(
        self,
        members: Tuple[int, ...],
        lo: np.ndarray,
        hi: np.ndarray,
        max_product_volume: float,
        max_weight: float,
        volumes_by_plant: np.ndarray,
        freight_by_q: np.ndarray,
    ) -> Optional[Cluster]:
        box = self.best_box(lo, hi, max_product_volume, max_weight)
        if box is None:
            return None
        pack = packaging_cost(volumes_by_plant, self.base_price)
        freight = float(freight_by_q[box.q])
        return Cluster(
            id=self._new_cluster_id(),
            members=members,
            lo=lo,
            hi=hi,
            max_product_volume=max_product_volume,
            max_weight=max_weight,
            volumes_by_plant=volumes_by_plant,
            freight_by_q=freight_by_q,
            q=box.q,
            dims=box.dims,
            cost_packaging=pack,
            cost_freight=freight,
            total_cost=pack + freight,
        )

    def singleton_clusters(self) -> Dict[int, Cluster]:
        clusters: Dict[int, Cluster] = {}
        for p in self.products:
            c = self._build_cluster_from_stats(
                members=(p.idx,),
                lo=self.product_lo[p.idx].copy(),
                hi=self.product_hi[p.idx].copy(),
                max_product_volume=p.orig_volume,
                max_weight=p.weight_kg,
                volumes_by_plant=p.volumes.copy(),
                freight_by_q=self.product_freight_by_q[p.idx].copy(),
            )
            if c is None:
                raise RuntimeError(f"No existe caja factible para el producto {p.code} con t={self.thickness}")
            clusters[c.id] = c
        return clusters

    def merge_clusters(self, a: Cluster, b: Cluster) -> Optional[Cluster]:
        lo = np.maximum(a.lo, b.lo)
        hi = np.minimum(a.hi, b.hi)
        if np.any(lo > hi + 1e-9):
            return None
        return self._build_cluster_from_stats(
            members=tuple(sorted(a.members + b.members)),
            lo=lo,
            hi=hi,
            max_product_volume=max(a.max_product_volume, b.max_product_volume),
            max_weight=max(a.max_weight, b.max_weight),
            volumes_by_plant=a.volumes_by_plant + b.volumes_by_plant,
            freight_by_q=a.freight_by_q + b.freight_by_q,
        )

    def optimize_greedy(self, min_saving: float = 1e-6, verbose: bool = True) -> List[Cluster]:
        clusters = self.singleton_clusters()
        heap: List[Tuple[float, int, int, int, int, Cluster]] = []

        active_ids = list(clusters.keys())
        total_pairs = len(active_ids) * (len(active_ids) - 1) // 2
        checked = 0
        for pos, aid in enumerate(active_ids):
            a = clusters[aid]
            for bid in active_ids[pos + 1 :]:
                b = clusters[bid]
                merged = self.merge_clusters(a, b)
                checked += 1
                if merged is None:
                    continue
                saving = a.total_cost + b.total_cost - merged.total_cost
                if saving > min_saving:
                    heapq.heappush(heap, (-saving, aid, bid, a.version, b.version, merged))
            if verbose and (pos + 1) % 50 == 0:
                print(f"  pares iniciales: {checked:,}/{total_pairs:,}")

        merges_done = 0
        saving_acc = 0.0
        while heap:
            neg_saving, aid, bid, aver, bver, candidate = heapq.heappop(heap)
            saving = -neg_saving
            a = clusters.get(aid)
            b = clusters.get(bid)
            if a is None or b is None or not a.active or not b.active:
                continue
            if a.version != aver or b.version != bver:
                continue

            # Recalcular para evitar aceptar un objeto candidato obsoleto por cualquier cambio futuro.
            merged = self.merge_clusters(a, b)
            if merged is None:
                continue
            real_saving = a.total_cost + b.total_cost - merged.total_cost
            if real_saving <= min_saving:
                continue

            a.active = False
            b.active = False
            a.version += 1
            b.version += 1
            clusters[merged.id] = merged
            merges_done += 1
            saving_acc += real_saving

            others = [c for c in clusters.values() if c.active and c.id != merged.id]
            for other in others:
                new_candidate = self.merge_clusters(merged, other)
                if new_candidate is None:
                    continue
                new_saving = merged.total_cost + other.total_cost - new_candidate.total_cost
                if new_saving > min_saving:
                    heapq.heappush(
                        heap,
                        (-new_saving, merged.id, other.id, merged.version, other.version, new_candidate),
                    )

            if verbose and (merges_done <= 10 or merges_done % 25 == 0):
                n_active = sum(1 for c in clusters.values() if c.active)
                print(
                    f"  merge {merges_done:3d}: ahorro marginal USD {real_saving:,.2f}; "
                    f"clusters activos={n_active}; ahorro acumulado={saving_acc:,.2f}"
                )

        result = sorted((c for c in clusters.values() if c.active), key=lambda c: c.id)
        return result

    def cluster_from_members(self, members: Iterable[int]) -> Optional[Cluster]:
        members_tuple = tuple(sorted(set(members)))
        if not members_tuple:
            return None
        idx = np.asarray(members_tuple, dtype=int)
        return self._build_cluster_from_stats(
            members=members_tuple,
            lo=self.product_lo[idx].max(axis=0),
            hi=self.product_hi[idx].min(axis=0),
            max_product_volume=max(self.products[i].orig_volume for i in members_tuple),
            max_weight=max(self.products[i].weight_kg for i in members_tuple),
            volumes_by_plant=np.sum([self.products[i].volumes for i in members_tuple], axis=0),
            freight_by_q=self.product_freight_by_q[idx].sum(axis=0),
        )

    def improve_local(
        self,
        clusters: Sequence[Cluster],
        min_saving: float = 1e-6,
        max_cycles: int = 10,
        verbose: bool = True,
    ) -> List[Cluster]:
        """Variable-neighborhood descent: relocaciones y merges con ahorro positivo."""
        current = list(clusters)
        cache: Dict[Tuple[int, ...], Optional[Cluster]] = {}

        def build(members: Iterable[int]) -> Optional[Cluster]:
            key = tuple(sorted(set(members)))
            if not key:
                return None
            if key not in cache:
                cache[key] = self.cluster_from_members(key)
            return cache[key]

        for cycle in range(max_cycles):
            changes = 0

            # Mejor relocalización de un producto por iteración.
            while True:
                membership = {i: j for j, c in enumerate(current) for i in c.members}
                best_delta = 0.0
                best_move = None
                for product_idx in range(len(self.products)):
                    source_idx = membership[product_idx]
                    source = current[source_idx]
                    source_without = (
                        build(i for i in source.members if i != product_idx)
                        if len(source.members) > 1
                        else None
                    )
                    source_without_cost = source_without.total_cost if source_without else 0.0

                    for target_idx, target in enumerate(current):
                        if target_idx == source_idx:
                            continue
                        target_with = build(target.members + (product_idx,))
                        if target_with is None:
                            continue
                        delta = (
                            source_without_cost
                            + target_with.total_cost
                            - source.total_cost
                            - target.total_cost
                        )
                        if delta < best_delta - min_saving:
                            best_delta = delta
                            best_move = (source_idx, target_idx, source_without, target_with, product_idx)

                    # También se permite separar el producto en una caja propia si reduce costo.
                    if len(source.members) > 1:
                        singleton = build((product_idx,))
                        assert singleton is not None
                        delta = source_without_cost + singleton.total_cost - source.total_cost
                        if delta < best_delta - min_saving:
                            best_delta = delta
                            best_move = (source_idx, None, source_without, singleton, product_idx)

                if best_move is None:
                    break

                source_idx, target_idx, source_without, target_with, product_idx = best_move
                if target_idx is None:
                    assert source_without is not None
                    current[source_idx] = source_without
                    current.append(target_with)
                elif source_without is None:
                    old = current
                    current = [
                        target_with if k == target_idx else c
                        for k, c in enumerate(old)
                        if k != source_idx
                    ]
                else:
                    current[source_idx] = source_without
                    current[target_idx] = target_with
                changes += 1
                if verbose:
                    print(
                        f"  relocalización {self.products[product_idx].code}: "
                        f"ahorro USD {-best_delta:,.2f}"
                    )

            # Nuevos merges pueden volverse atractivos después de las relocalizaciones.
            while True:
                best_delta = 0.0
                best_merge = None
                for i, a in enumerate(current):
                    for j in range(i + 1, len(current)):
                        b = current[j]
                        merged = build(a.members + b.members)
                        if merged is None:
                            continue
                        delta = merged.total_cost - a.total_cost - b.total_cost
                        if delta < best_delta - min_saving:
                            best_delta = delta
                            best_merge = (i, j, merged)
                if best_merge is None:
                    break
                i, j, merged = best_merge
                current = [c for k, c in enumerate(current) if k not in (i, j)] + [merged]
                changes += 1
                if verbose:
                    print(f"  merge local: ahorro USD {-best_delta:,.2f}")

            if verbose:
                print(
                    f"  ciclo local {cycle + 1}: cambios={changes}, "
                    f"costo USD {sum(c.total_cost for c in current):,.2f}"
                )
            if changes == 0:
                break

        return current

    @staticmethod
    def physical_type_key(cluster: Cluster, decimals: int = 6) -> Tuple[float, float, float]:
        return tuple(round(float(x), decimals) for x in cluster.dims)

    def physical_type_count(self, clusters: Sequence[Cluster], decimals: int = 6) -> int:
        return len({self.physical_type_key(c, decimals) for c in clusters})

    def validate_cluster(self, cluster: Cluster) -> List[str]:
        errors: List[str] = []
        dims = np.asarray(cluster.dims, dtype=float)
        ext = dims + 2 * self.thickness
        n_l = int(math.floor((PALLET_ANCHO + 1e-9) / ext[0]))
        n_w = int(math.floor((PALLET_LARGO + 1e-9) / ext[1]))
        n_h = int(math.floor((PALLET_ALTO + 1e-9) / ext[2]))
        if n_l * n_w * n_h != cluster.q:
            errors.append(f"q inconsistente: almacenado={cluster.q}, recalculado={n_l*n_w*n_h}")

        for idx in cluster.members:
            p = self.products[idx]
            orig = np.asarray(p.orig_dims)
            if np.any(dims < 0.90 * orig - 1e-6) or np.any(dims > 1.10 * orig + 1e-6):
                errors.append(f"{p.code}: viola fit ±10%")
            if float(np.prod(dims)) + 1e-5 < p.orig_volume:
                errors.append(f"{p.code}: volumen interno insuficiente")
            headspace = dims - orig
            max_headspace = np.minimum(self.headspace_ratio * dims, 40.0)
            if np.any(headspace > max_headspace + 1e-6):
                errors.append(f"{p.code}: viola headspace")
            load_max = self.ect * (2.0 * (ext[0] + ext[1]) / 1000.0) / G
            required = max(0, n_h - 1) * p.weight_kg
            if load_max + 1e-6 < required:
                errors.append(f"{p.code}: viola ECT ({load_max:.2f} < {required:.2f} kg)")
        return errors


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_products(
    catalog_path: Path,
    specs_path: Path,
    operations_path: Path,
) -> Tuple[List[Product], Dict[str, Dict[str, str]], List[Dict[str, str]]]:
    specs_rows = read_csv_dicts(specs_path)
    specs = {row["caja_tipo_id"]: row for row in specs_rows}
    catalog_rows = read_csv_dicts(catalog_path)
    operations_rows = read_csv_dicts(operations_path)
    operations = {row["codigo_producto"]: row for row in operations_rows}

    products: List[Product] = []
    for idx, row in enumerate(catalog_rows):
        code = row["codigo_producto"]
        box_id = row["caja_tipo_id"]
        if box_id not in specs:
            raise KeyError(f"Caja {box_id} de {code} no aparece en especificaciones")
        if code not in operations:
            raise KeyError(f"Producto {code} no aparece en operaciones")
        spec = specs[box_id]
        op = operations[code]
        dims = (
            parse_float(spec["caja_interior_largo"]),
            parse_float(spec["caja_interior_ancho"]),
            parse_float(spec["caja_interior_alto"]),
        )
        volumes = np.array(
            [parse_float(op[f"volumen_producto_planta_{plant}"]) for plant in PLANTS],
            dtype=float,
        )
        products.append(
            Product(
                idx=idx,
                code=code,
                current_box_id=box_id,
                orig_dims=dims,
                weight_kg=parse_float(row["peso_neto_caja"]),
                volumes=volumes,
            )
        )
    return products, specs, operations_rows


def official_baseline(operations_rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    packaging = sum(parse_float(row["costo_total"]) for row in operations_rows)
    freight = sum(parse_float(row["costo_pallets_total"]) for row in operations_rows)
    return {"packaging": packaging, "freight": freight, "total": packaging + freight}


def summarize_solution(
    optimizer: BonsaiOptimizer,
    clusters: Sequence[Cluster],
    baseline: Dict[str, float],
) -> Dict[str, float]:
    packaging = sum(c.cost_packaging for c in clusters)
    freight = sum(c.cost_freight for c in clusters)
    total = packaging + freight
    errors = []
    for c in clusters:
        errors.extend(optimizer.validate_cluster(c))
    if errors:
        raise RuntimeError("Errores de validación:\n" + "\n".join(errors[:30]))

    return {
        "thickness": optimizer.thickness,
        "n_clusters": len(clusters),
        "n_types": optimizer.physical_type_count(clusters),
        "packaging": packaging,
        "freight": freight,
        "total": total,
        "saving": baseline["total"] - total,
        "saving_pct": 100.0 * (baseline["total"] - total) / baseline["total"],
    }


def write_assignment_csv(
    path: Path,
    optimizer: BonsaiOptimizer,
    clusters: Sequence[Cluster],
) -> None:
    assignment: Dict[int, Cluster] = {}
    for c in clusters:
        for idx in c.members:
            if idx in assignment:
                raise RuntimeError(f"Producto duplicado en clusters: {idx}")
            assignment[idx] = c
    if len(assignment) != len(optimizer.products):
        raise RuntimeError("No todos los productos quedaron asignados")

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
        for p in optimizer.products:
            c = assignment[p.idx]
            l, w, h = c.dims
            writer.writerow(
                {
                    "codigo_producto": p.code,
                    "caja_grosor_mm": f"{optimizer.thickness:.1f}",
                    "caja_exterior_largo": f"{l + 2 * optimizer.thickness:.9f}",
                    "caja_exterior_ancho": f"{w + 2 * optimizer.thickness:.9f}",
                    "caja_exterior_alto": f"{h + 2 * optimizer.thickness:.9f}",
                }
            )


def write_clusters_csv(
    path: Path,
    optimizer: BonsaiOptimizer,
    clusters: Sequence[Cluster],
) -> None:
    # La submission define el tipo físico por grosor + dimensiones. Clusters distintos
    # con dimensiones idénticas se reportan como un único tipo.
    grouped: Dict[Tuple[float, float, float], List[Cluster]] = {}
    for cluster in clusters:
        grouped.setdefault(optimizer.physical_type_key(cluster), []).append(cluster)

    records = []
    for key, group in grouped.items():
        representative = group[0]
        members = tuple(sorted(i for c in group for i in c.members))
        volumes = np.sum([c.volumes_by_plant for c in group], axis=0)
        pack = packaging_cost(volumes, optimizer.base_price)
        freight = sum(c.cost_freight for c in group)
        records.append((members, representative, pack, freight))

    records.sort(key=lambda x: (-len(x[0]), x[0]))
    with path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "tipo_propuesto",
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
            "costo_packaging",
            "costo_flete",
            "costo_total",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for j, (members, c, pack, freight) in enumerate(records, start=1):
            l, w, h = c.dims
            writer.writerow(
                {
                    "tipo_propuesto": f"BOX_{j:03d}",
                    "n_productos": len(members),
                    "productos": "|".join(optimizer.products[i].code for i in members),
                    "grosor_mm": f"{optimizer.thickness:.1f}",
                    "interior_largo": f"{l:.9f}",
                    "interior_ancho": f"{w:.9f}",
                    "interior_alto": f"{h:.9f}",
                    "exterior_largo": f"{l + 2 * optimizer.thickness:.9f}",
                    "exterior_ancho": f"{w + 2 * optimizer.thickness:.9f}",
                    "exterior_alto": f"{h + 2 * optimizer.thickness:.9f}",
                    "cajas_por_pallet": c.q,
                    "costo_packaging": f"{pack:.2f}",
                    "costo_flete": f"{freight:.2f}",
                    "costo_total": f"{pack + freight:.2f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/data"))
    parser.add_argument("--catalog", default="catalogo_productos(1).csv")
    parser.add_argument("--specs", default="especificaciones_cajas(2).csv")
    parser.add_argument("--operations", default="operaciones_planta(2).csv")
    parser.add_argument("--thickness", nargs="+", type=float, default=[3.0, 4.5, 5.0])
    parser.add_argument("--pallet-cost", type=float, default=150.0)
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/data/resultados_bonsai"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-local-search", action="store_true")
    args = parser.parse_args()

    products, _specs, operations_rows = load_products(
        args.data_dir / args.catalog,
        args.data_dir / args.specs,
        args.data_dir / args.operations,
    )
    baseline = official_baseline(operations_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Productos: {len(products)}")
    print(
        "Baseline oficial Operations: "
        f"packaging USD {baseline['packaging']:,.2f}, "
        f"flete USD {baseline['freight']:,.2f}, total USD {baseline['total']:,.2f}"
    )

    all_summaries = []
    for thickness in args.thickness:
        print(f"\n=== Grosor uniforme {thickness:.1f} mm ===")
        optimizer = BonsaiOptimizer(products, thickness, pallet_costs=args.pallet_cost)
        clusters = optimizer.optimize_greedy(verbose=not args.quiet)
        if not args.skip_local_search:
            clusters = optimizer.improve_local(clusters, verbose=not args.quiet)
        summary = summarize_solution(optimizer, clusters, baseline)
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        label = str(thickness).replace(".", "_")
        write_assignment_csv(args.output_dir / f"submission_t{label}.csv", optimizer, clusters)
        write_clusters_csv(args.output_dir / f"clusters_t{label}.csv", optimizer, clusters)

    all_summaries.sort(key=lambda x: x["total"])
    with (args.output_dir / "resumen_escenarios.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)

    best = all_summaries[0]
    print("\n=== Mejor escenario greedy ===")
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
