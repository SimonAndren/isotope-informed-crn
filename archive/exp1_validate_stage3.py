#!/usr/bin/env python
"""Experiment 1: Validate Stage 3 analysis code using Goldman's DRG mechanism.

Goldman's DRG YAML (31 species, 167 reactions) gives Table 2 ≈ 0.6 via
compute_slopes_goldman() + get_delta() + cluster CSV.

This script runs the SAME mechanism through get_delta_ours() with a
DRG_SPECIES_MAP (analogous to OUR_SPECIES_MAP) to verify our analysis
code path is correct.

If the result ≈ 0.6, our analysis code is fine and the bug is in Stage 2.
If the result ≈ 7+, there's a bug in get_delta_ours / our_init_x / species map.
"""
from __future__ import annotations

import pathlib
import sys
import time

import cantera as ct
import numpy as np
import pandas as pd
from scipy.stats import linregress

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from isotopologue.goldman import VPDB_RATIO, get_delta, run_simulation, enrich_frac

_ROOT = pathlib.Path(__file__).parents[1]
_GOLDMAN_DIR = _ROOT / "rmg_inputs" / "goldmanm-RMG_isotopes_paper_data-234bd52"
_EXP_DATA = _GOLDMAN_DIR / "exp_data"
_DRG_DIR = _GOLDMAN_DIR / "mechanisms" / "drg_model"

TEMPERATURES = [800, 850, 900, 950]
PSIA_VALUES = np.linspace(-10, 20, 5)

RELATIONSHIPS = [
    ("dC2H4 = f(dCH4)", "methane", "ethene"),
    ("dC2H6 = f(dCH4)", "methane", "ethane"),
    ("dC2H6 = f(dC2H4)", "ethene", "ethane"),
    ("dBulk = f(dCH4)", "methane", "bulk"),
]

# DRG cluster numbers (from isotopomer_cluster_info.csv)
DRG_CLUSTERS = {"propane": 8, "methane": 1, "ethene": 2, "ethane": 7}

# DRG_SPECIES_MAP: species_name -> (cluster_num, n_13C, total_C)
# Built directly from the cluster CSV enriched_atoms / unenriched_atoms columns
DRG_SPECIES_MAP: dict[str, tuple[int, int, int]] = {
    # Methane cluster 1 (total_C=1)
    "C":   (1, 0, 1),
    "C-2": (1, 1, 1),
    # Ethylene cluster 2 (total_C=2)
    "C=C":   (2, 0, 2),
    "C=C-2": (2, 2, 2),
    "C=C-3": (2, 1, 2),
    # Ethane cluster 7 (total_C=2)
    "CC":   (7, 0, 2),
    "CC-2": (7, 2, 2),
    "CC-3": (7, 1, 2),
    # Propane cluster 8 (total_C=3)
    "CCC":   (8, 0, 3),
    "CCC-2": (8, 3, 3),
    "CCC-3": (8, 2, 3),
    "CCC-4": (8, 2, 3),
    "CCC-5": (8, 1, 3),
    "CCC-6": (8, 1, 3),
}


def get_delta_with_map(
    concentrations: pd.Series | dict,
    species_map: dict[str, tuple[int, int, int]],
    cluster_num: int,
    ref_ratio: float = VPDB_RATIO,
) -> float:
    """Same logic as get_delta_ours but with configurable species map."""
    numer = denom = 0.0
    for sp, (cn, n13, total_c) in species_map.items():
        if cn != cluster_num:
            continue
        c = concentrations[sp] if isinstance(concentrations, dict) else concentrations.get(sp, 0.0)
        numer += c * n13
        denom += c * (total_c - n13)
    if denom <= 0 or np.isclose(denom, 0.0):
        return np.nan
    return (numer / denom / ref_ratio - 1.0) * 1000.0


def drg_init_x(psia: float, fraction_propane: float = 0.0049, center_delta: float = -28.0) -> dict:
    """Goldman DRG species names (same as goldman_init_x)."""
    e = enrich_frac(center_delta + psia)
    c = enrich_frac(center_delta)
    f = fraction_propane
    return {
        "CCC":   f * (1 - c) * (1 - e) ** 2,
        "CCC-5": f * 2 * e * (1 - c) * (1 - e),
        "CCC-6": f * c * (1 - e) ** 2,
        "CCC-3": f * (1 - c) * e ** 2,
        "CCC-4": f * 2 * e * (1 - e) * c,
        "CCC-2": f * c * e ** 2,
        "[He]":  1.0 - f,
    }


def _make_slopes_df() -> pd.DataFrame:
    return pd.DataFrame(
        index=[r[0] for r in RELATIONSHIPS],
        columns=TEMPERATURES,
        dtype=float,
    )


def _fit_slopes(rows: list[dict], slopes: pd.DataFrame, T_C: int) -> None:
    enrich = pd.DataFrame(rows, index=PSIA_VALUES)
    for label, x_mol, y_mol in RELATIONSHIPS:
        x = enrich[x_mol].values
        y = enrich[y_mol].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 2:
            slope, *_ = linregress(x[mask], y[mask])
            slopes.loc[label, T_C] = slope


def compute_slopes_drg_via_csv(yaml_path, csv_path, clusters, init_x_fn):
    """Goldman's code path: get_delta with cluster CSV."""
    ci = pd.read_csv(csv_path, index_col="name")
    slopes = _make_slopes_df()
    for T_C in TEMPERATURES:
        T_K, t_final = T_C + 273.15, 95.0 / T_C
        rows = []
        for psia in PSIA_VALUES:
            gas = ct.Solution(str(yaml_path))
            init_x = init_x_fn(psia=psia)
            df = run_simulation(gas, [t_final], T_K=T_K, P_Pa=2e5, init_x=init_x)
            final = df.iloc[0]
            row = {mol: get_delta(final, ci, cluster_num=clusters[mol])
                   for mol in ("methane", "ethene", "ethane")}
            row["bulk"] = get_delta(pd.Series(init_x), ci, cluster_num=clusters["propane"])
            rows.append(row)
        _fit_slopes(rows, slopes, T_C)
    return slopes


def compute_slopes_drg_via_map(yaml_path, species_map, clusters, init_x_fn):
    """Our code path: get_delta_with_map (same logic as get_delta_ours)."""
    slopes = _make_slopes_df()
    for T_C in TEMPERATURES:
        T_K, t_final = T_C + 273.15, 95.0 / T_C
        rows = []
        for psia in PSIA_VALUES:
            gas = ct.Solution(str(yaml_path))
            init_x = init_x_fn(psia=psia)
            df = run_simulation(gas, [t_final], T_K=T_K, P_Pa=2e5, init_x=init_x)
            final = df.iloc[0]
            row = {mol: get_delta_with_map(final, species_map, cluster_num=clusters[mol])
                   for mol in ("methane", "ethene", "ethane")}
            row["bulk"] = get_delta_with_map(init_x, species_map, cluster_num=clusters["propane"])
            rows.append(row)
        _fit_slopes(rows, slopes, T_C)
    return slopes


def compute_table2_score(slopes, gilbert_values, gilbert_unc):
    scaled = (gilbert_values - slopes) / gilbert_unc
    vals = scaled.values[np.isfinite(scaled.values)]
    return np.std(vals) if len(vals) > 0 else float("nan")


def main():
    yaml_path = _DRG_DIR / "chem.yaml"
    csv_path = _DRG_DIR / "isotopomer_cluster_info.csv"

    gilbert_values = pd.read_csv(_EXP_DATA / "Gilbert_table2_values.csv", index_col="relationship")
    gilbert_unc = pd.read_csv(_EXP_DATA / "Gilbert_table2_uncertainty.csv", index_col="relationship")
    gilbert_values.columns = [int(c) for c in gilbert_values.columns]
    gilbert_unc.columns = [int(c) for c in gilbert_unc.columns]

    # --- Approach A: Goldman's code path (cluster CSV) ---
    print("=" * 60)
    print("Approach A: get_delta() with cluster CSV (Goldman's path)")
    print("=" * 60)
    t0 = time.perf_counter()
    slopes_csv = compute_slopes_drg_via_csv(yaml_path, csv_path, DRG_CLUSTERS, drg_init_x)
    elapsed_a = time.perf_counter() - t0
    print(slopes_csv.to_string(float_format=lambda x: f"{x:.4f}"))
    score_a = compute_table2_score(slopes_csv, gilbert_values, gilbert_unc)
    print(f"\nTable 2 score: {score_a:.2f}  (expected ~0.6)")
    print(f"Elapsed: {elapsed_a:.1f}s")

    # --- Approach B: Our code path (species map) ---
    print(f"\n{'=' * 60}")
    print("Approach B: get_delta_with_map() using DRG_SPECIES_MAP (our path)")
    print("=" * 60)
    t0 = time.perf_counter()
    slopes_map = compute_slopes_drg_via_map(yaml_path, DRG_SPECIES_MAP, DRG_CLUSTERS, drg_init_x)
    elapsed_b = time.perf_counter() - t0
    print(slopes_map.to_string(float_format=lambda x: f"{x:.4f}"))
    score_b = compute_table2_score(slopes_map, gilbert_values, gilbert_unc)
    print(f"\nTable 2 score: {score_b:.2f}  (expected ~0.6)")
    print(f"Elapsed: {elapsed_b:.1f}s")

    # --- Compare ---
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print("=" * 60)
    diff = slopes_csv - slopes_map
    print("Slope differences (CSV - MAP):")
    print(diff.to_string(float_format=lambda x: f"{x:.6f}"))
    max_diff = np.nanmax(np.abs(diff.values.astype(float)))
    print(f"\nMax absolute slope difference: {max_diff:.6f}")
    print(f"\nTable 2 scores:")
    print(f"  Approach A (CSV):  {score_a:.2f}")
    print(f"  Approach B (MAP):  {score_b:.2f}")
    print(f"  Goldman ref (DRG): 0.6")

    if np.isclose(score_a, score_b, atol=0.05) and score_a < 1.0:
        print("\nRESULT: Both approaches agree and match Goldman. Analysis code is CORRECT.")
    elif not np.isclose(score_a, score_b, atol=0.05):
        print(f"\nRESULT: Approaches DISAGREE by {abs(score_a - score_b):.2f}. "
              "Bug in get_delta_ours / species map!")
    elif score_a > 1.0:
        print(f"\nRESULT: Both approaches give wrong score ({score_a:.1f}). "
              "Issue is elsewhere (not in species map).")


if __name__ == "__main__":
    main()
