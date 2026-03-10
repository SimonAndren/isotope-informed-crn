#!/usr/bin/env python
"""Compare our generated mechanism's δ¹³C output to Goldman 2019 Table 2.

Runs the same PSIA × temperature sweep as replicate_table2.py but using
our freshly-generated isotopologue mechanism instead of Goldman's pre-built one.

Usage:
    uv run python rmg_pipeline/compare_to_goldman.py

Species name mapping (our CHEMKIN → Goldman SMILES):
    propane_ooo(1) ↔ CCC    (0 ¹³C)
    CCC(5)         ↔ CCC-5  (1 edge ¹³C, ×2 symmetry)
    CCC(6)         ↔ CCC-6  (center ¹³C)
    CCC(3)         ↔ CCC-3  (both edges ¹³C)
    CCC(4)         ↔ CCC-4  (center + 1 edge ¹³C, ×2 symmetry)
    CCC(2)         ↔ CCC-2  (all 3 ¹³C)

OUR_SPECIES_MAP entries derived from species_dictionary.txt by counting
`i13` labels per species (carbon-13 label used by RMG isotopologue tool).
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

_ROOT    = pathlib.Path(__file__).parents[1]
_OUR_DIR = _ROOT / "rmg_pipeline" / "output" / "isotope" / "iso"
_GOLDMAN_DIR = _ROOT / "rmg_inputs" / "goldmanm-RMG_isotopes_paper_data-234bd52"
_EXP_DATA    = _GOLDMAN_DIR / "exp_data"

TEMPERATURES = [800, 850, 900, 950]   # °C
PSIA_VALUES  = np.linspace(-10, 20, 5)

RELATIONSHIPS = [
    ("dC2H4 = f(dCH4)",  "methane", "ethene"),
    ("dC2H6 = f(dCH4)",  "methane", "ethane"),
    ("dC2H6 = f(dC2H4)", "ethene",  "ethane"),
    ("dBulk = f(dCH4)",  "methane", "bulk"),
]

# Cluster numbers in Goldman's full model
GOLDMAN_CLUSTERS = {"propane": 26, "methane": 19, "ethene": 22, "ethane": 18}

# Cluster numbers in our generated mechanism (from isotopomer_cluster_info.csv)
OUR_CLUSTERS = {"propane": 26, "methane": 16, "ethene": 22, "ethane": 15}

# Explicit mapping: our YAML name → (cluster_num, n_13C, total_C)
# n_13C verified by counting `i13` in chemkin/species_dictionary.txt.
# Cluster numbers verified against isotopomer_cluster_info.csv.
# NOTE: Species indices change when Stage 1 runs inline (use_original_reactions=True)
# vs loaded from saved CHEMKIN. Update these after each Stage 2 re-run.
OUR_SPECIES_MAP: dict[str, tuple[int, int, int]] = {
    # Methane (cluster 16, total_C=1)
    "C(46)": (16, 0, 1),
    "C(47)": (16, 1, 1),
    # Ethylene (cluster 22, total_C=2)
    "C2H4(14)": (22, 0, 2),
    "C2H4(15)": (22, 2, 2),
    "C2H4(16)": (22, 1, 2),
    # Ethane (cluster 15, total_C=2)
    "CC(48)": (15, 0, 2),
    "CC(49)": (15, 2, 2),
    "CC(50)": (15, 1, 2),
    # Propane (cluster 26, total_C=3)
    "propane_ooo(1)": (26, 0, 3),
    "CCC(2)":         (26, 3, 3),
    "CCC(3)":         (26, 2, 3),
    "CCC(4)":         (26, 2, 3),
    "CCC(5)":         (26, 1, 3),
    "CCC(6)":         (26, 1, 3),
}


def get_delta_ours(
    concentrations: pd.Series | dict,
    cluster_num: int,
    ref_ratio: float = VPDB_RATIO,
) -> float:
    """δ¹³C (‰) for a cluster in our mechanism.

    Uses OUR_SPECIES_MAP to look up n_13C and total_C per species.
    Avoids the duplicate-name problem in our cluster CSV (which stores
    base SMILES without isotope labels, unlike Goldman's unique-SMILES CSV).
    """
    numer = denom = 0.0
    for sp, (cn, n13, total_c) in OUR_SPECIES_MAP.items():
        if cn != cluster_num:
            continue
        c = concentrations[sp] if isinstance(concentrations, dict) else concentrations.get(sp, 0.0)
        numer += c * n13
        denom += c * (total_c - n13)   # 12C atoms only, matching get_delta's unenriched_atoms
    if denom <= 0 or np.isclose(denom, 0.0):
        return np.nan
    return (numer / denom / ref_ratio - 1.0) * 1000.0


def our_init_x(psia: float, fraction_propane: float = 0.0049, center_delta: float = -28.0) -> dict:
    """table2_init_x with our CHEMKIN species names instead of Goldman's SMILES."""
    e = enrich_frac(center_delta + psia)   # edge δ¹³C = center + psia
    c = enrich_frac(center_delta)           # center fixed
    f = fraction_propane
    return {
        "propane_ooo(1)": f * (1 - c) * (1 - e) ** 2,
        "CCC(5)":         f * 2 * e * (1 - c) * (1 - e),   # one edge (×2 for C1/C3 symmetry)
        "CCC(6)":         f * c * (1 - e) ** 2,
        "CCC(3)":         f * (1 - c) * e ** 2,
        "CCC(4)":         f * 2 * e * (1 - e) * c,          # center+edge (×2 symmetry)
        "CCC(2)":         f * c * e ** 2,
        "He":             1.0 - f,
    }


def goldman_init_x(psia: float, fraction_propane: float = 0.0049, center_delta: float = -28.0) -> dict:
    """Same as our_init_x but with Goldman's SMILES species names."""
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


def compute_slopes_goldman(yaml_path: pathlib.Path, csv_path: pathlib.Path,
                           clusters: dict, init_x_fn) -> pd.DataFrame:
    """PSIA × T sweep for Goldman's mechanism (uses cluster CSV directly)."""
    ci = pd.read_csv(csv_path, index_col="name")
    slopes = _make_slopes_df()

    for T_C in TEMPERATURES:
        T_K, t_final = T_C + 273.15, 95.0 / T_C
        rows = []
        for psia in PSIA_VALUES:
            gas    = ct.Solution(str(yaml_path))
            init_x = init_x_fn(psia=psia)
            t0 = time.perf_counter()
            df = run_simulation(gas, [t_final], T_K=T_K, P_Pa=2e5, init_x=init_x)
            elapsed = time.perf_counter() - t0
            final = df.iloc[0]
            row = {mol: get_delta(final, ci, cluster_num=clusters[mol])
                   for mol in ("methane", "ethene", "ethane")}
            row["bulk"] = get_delta(pd.Series(init_x), ci, cluster_num=clusters["propane"])
            rows.append(row)
            print(f"  T={T_C}°C  psia={psia:+5.1f}"
                  f"  δCH4={row['methane']:+6.2f}  δC2H4={row['ethene']:+6.2f}"
                  f"  ({elapsed:.2f}s)")
        _fit_slopes(rows, slopes, T_C)
    return slopes


def compute_slopes_ours(yaml_path: pathlib.Path, init_x_fn) -> pd.DataFrame:
    """PSIA × T sweep for our mechanism (uses OUR_SPECIES_MAP for δ¹³C)."""
    slopes = _make_slopes_df()

    for T_C in TEMPERATURES:
        T_K, t_final = T_C + 273.15, 95.0 / T_C
        rows = []
        for psia in PSIA_VALUES:
            gas    = ct.Solution(str(yaml_path))
            init_x = init_x_fn(psia=psia)
            t0 = time.perf_counter()
            df = run_simulation(gas, [t_final], T_K=T_K, P_Pa=2e5, init_x=init_x)
            elapsed = time.perf_counter() - t0
            final = df.iloc[0]
            # Use explicit species map — avoids duplicate-name ambiguity in our cluster CSV
            row = {mol: get_delta_ours(final, cluster_num=OUR_CLUSTERS[mol])
                   for mol in ("methane", "ethene", "ethane")}
            row["bulk"] = get_delta_ours(init_x, cluster_num=OUR_CLUSTERS["propane"])
            rows.append(row)
            print(f"  T={T_C}°C  psia={psia:+5.1f}"
                  f"  δCH4={row['methane']:+6.2f}  δC2H4={row['ethene']:+6.2f}"
                  f"  ({elapsed:.2f}s)")
        _fit_slopes(rows, slopes, T_C)
    return slopes


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


def main() -> None:
    gilbert_values = pd.read_csv(_EXP_DATA / "Gilbert_table2_values.csv",     index_col="relationship")
    gilbert_unc    = pd.read_csv(_EXP_DATA / "Gilbert_table2_uncertainty.csv", index_col="relationship")
    gilbert_values.columns = [int(c) for c in gilbert_values.columns]
    gilbert_unc.columns    = [int(c) for c in gilbert_unc.columns]

    results: dict[str, float] = {}
    slopes_all: dict[str, pd.DataFrame] = {}
    t_total = time.perf_counter()

    # --- Our model ---
    print(f"\n{'='*60}")
    print("Model: ours")
    print("=" * 60)
    t0 = time.perf_counter()
    s = compute_slopes_ours(_OUR_DIR / "chem.yaml", our_init_x)
    slopes_all["ours"] = s
    print(f"\n  Fitted slopes (ours):")
    print(s.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"  → done in {time.perf_counter() - t0:.1f}s")
    scaled = (gilbert_values - s) / gilbert_unc
    results["ours"] = np.std(scaled.values[np.isfinite(scaled.values)])

    # --- Goldman full model ---
    print(f"\n{'='*60}")
    print("Model: goldman-full")
    print("=" * 60)
    t0 = time.perf_counter()
    s = compute_slopes_goldman(
        _GOLDMAN_DIR / "mechanisms" / "full_model" / "chem.yaml",
        _GOLDMAN_DIR / "mechanisms" / "full_model" / "isotopomer_cluster_info.csv",
        GOLDMAN_CLUSTERS,
        goldman_init_x,
    )
    slopes_all["goldman-full"] = s
    print(f"\n  Fitted slopes (goldman-full):")
    print(s.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"  → done in {time.perf_counter() - t0:.1f}s")
    scaled = (gilbert_values - s) / gilbert_unc
    results["goldman-full"] = np.std(scaled.values[np.isfinite(scaled.values)])

    print(f"\n{'='*60}")
    print("Table 2: std(scaled deviations from Gilbert 2016)")
    print("=" * 60)
    table = pd.DataFrame(
        {name: [round(v, 1)] for name, v in results.items()},
        index=["all"],
    )
    print(table.to_string())

    print()
    print("Goldman 2019 reference (Table 2):")
    print("  model    full   drg  6rxn  3rxn")
    print("  all       0.5   0.6   6.5   1.7")
    print(f"\nTotal elapsed: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
