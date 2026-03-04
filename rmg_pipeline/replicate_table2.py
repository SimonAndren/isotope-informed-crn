#!/usr/bin/env python
"""Replicate Goldman 2019 Table 2: scaled deviations from Gilbert 2016.

Sweeps PSIA across 5 values at 4 temperatures for each mechanism model.
At each condition, simulates to t_final = 95/T_K, computes δ¹³C for
methane / ethene / ethane / bulk, fits enrichment slopes, and compares
to Gilbert et al. 2016 experimental data.

Usage:
    uv run python rmg_pipeline/replicate_table2.py

Expected (Goldman 2019 Table 2):
    model   full  drg  6rxn  3rxn
    all      0.5  0.6   6.5   1.7

The 3-rxn and 6-rxn models finish in seconds; DRG takes a few minutes;
the full model (343 sp / 7096 rxn) takes ~20-40 minutes.
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

from isotopologue.goldman import get_delta, table2_init_x, run_simulation

_BASE = pathlib.Path(__file__).parents[1] / (
    "rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52"
)
_EXP_DATA = _BASE / "exp_data"

# Cluster number for each molecule, per model (from Goldman's create_paper_figures.py)
CLUSTERS = {
    "3rxn": {"propane": 5, "methane": 0, "ethene": 1, "ethane": 4},
    "6rxn": {"propane": 7, "methane": 1, "ethene": 2, "ethane": 6},
    "drg":  {"propane": 8, "methane": 1, "ethene": 2, "ethane": 7},
    "full": {"propane": 26, "methane": 19, "ethene": 22, "ethane": 18},
}

MECHANISMS = [
    ("3rxn", "three_reaction_model"),
    ("6rxn", "six_reaction_model"),
    ("drg",  "drg_model"),
    ("full", "full_model"),
]

TEMPERATURES = [800, 850, 900, 950]   # °C
PSIA_VALUES  = np.linspace(-10, 20, 5)

# Table 2 relationships: (label, x_molecule, y_molecule)
RELATIONSHIPS = [
    ("dC2H4 = f(dCH4)",  "methane", "ethene"),
    ("dC2H6 = f(dCH4)",  "methane", "ethane"),
    ("dC2H6 = f(dC2H4)", "ethene",  "ethane"),
    ("dBulk = f(dCH4)",  "methane", "bulk"),
]


def _model_slopes(name: str, model_dir: pathlib.Path) -> pd.DataFrame:
    """Run all psia × temperature simulations and return fitted slopes.

    Returns DataFrame indexed by relationship label, columns = temperatures.
    """
    ci = pd.read_csv(model_dir / "isotopomer_cluster_info.csv", index_col="name")
    clusters = CLUSTERS[name]

    slopes = pd.DataFrame(
        index=[r[0] for r in RELATIONSHIPS],
        columns=TEMPERATURES,
        dtype=float,
    )

    for T_C in TEMPERATURES:
        T_K     = T_C + 273.15
        t_final = 95.0 / T_C   # Goldman Table 2 uses T_Celsius (see supplemental)

        # Accumulate enrichments: rows = psia values, cols = molecules
        rows = []
        for psia in PSIA_VALUES:
            gas    = ct.Solution(str(model_dir / "chem.yaml"))
            init_x = table2_init_x(psia=psia)

            t0 = time.perf_counter()
            df = run_simulation(gas, [t_final], T_K=T_K, P_Pa=2e5, init_x=init_x)
            elapsed = time.perf_counter() - t0

            final = df.iloc[0]
            row = {
                mol: get_delta(final, ci, cluster_num=clusters[mol])
                for mol in ("methane", "ethene", "ethane")
            }
            # Bulk = δ¹³C of initial propane (before any reaction)
            row["bulk"] = get_delta(pd.Series(init_x), ci, cluster_num=clusters["propane"])

            rows.append(row)
            print(
                f"  {name:4s}  T={T_C}°C  psia={psia:+5.1f}"
                f"  δCH4={row['methane']:+6.2f}  δC2H4={row['ethene']:+6.2f}"
                f"  {elapsed:.2f}s"
            )

        enrich = pd.DataFrame(rows, index=PSIA_VALUES)   # (psia × molecule)

        for label, x_mol, y_mol in RELATIONSHIPS:
            x = enrich[x_mol].values
            y = enrich[y_mol].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2:
                slope, *_ = linregress(x[mask], y[mask])
                slopes.loc[label, T_C] = slope

    return slopes


def main() -> None:
    gilbert_values = pd.read_csv(
        _EXP_DATA / "Gilbert_table2_values.csv", index_col="relationship"
    )
    gilbert_values.columns = [int(c) for c in gilbert_values.columns]
    gilbert_unc = pd.read_csv(
        _EXP_DATA / "Gilbert_table2_uncertainty.csv", index_col="relationship"
    )
    gilbert_unc.columns = [int(c) for c in gilbert_unc.columns]

    model_slopes: dict[str, pd.DataFrame] = {}
    t_total = time.perf_counter()

    for name, subdir in MECHANISMS:
        model_dir = _BASE / "mechanisms" / subdir
        print(f"\n{'='*60}")
        print(f"Model: {name}  ({subdir})")
        print("=" * 60)
        t0 = time.perf_counter()
        model_slopes[name] = _model_slopes(name, model_dir)
        print(f"  → {name} done in {time.perf_counter() - t0:.1f}s")

        # Print partial slopes for this model
        print(f"\n  Fitted slopes ({name}):")
        print(model_slopes[name].to_string(float_format=lambda x: f"{x:.3f}"))

    # ── Table 2 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Table 2: std(scaled deviations from Gilbert 2016)")
    print("=" * 60)

    results = {}
    for name, slopes in model_slopes.items():
        scaled = (gilbert_values - slopes) / gilbert_unc
        results[name] = np.std(scaled.values.flatten())

    col_order = ["full", "drg", "6rxn", "3rxn"]
    available = [c for c in col_order if c in results]
    table = pd.DataFrame(
        {name: [round(results[name], 1)] for name in available},
        index=["all"],
    )
    print(table.to_string())
    print()
    print("Goldman 2019 reference:")
    print("         full  drg   6rxn  3rxn")
    print("  all     0.5  0.6    6.5   1.7")
    print(f"\nTotal elapsed: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
