#!/usr/bin/env python
"""Replicate Goldman 2019 Figure 6 using both our mechanism and Goldman's full model.

Plots δ¹³C of products vs δ¹³C of CH4 for each relationship across the PSIA sweep,
with one subplot per relationship and colours per temperature.

Also prints raw ¹³C/¹²C ratios to check whether the VPDB reference ratio could
explain the observed δ offset (it cannot — slopes are reference-invariant).

Usage:
    uv run python rmg_pipeline/plot_figure6.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from isotopologue.goldman import VPDB_RATIO, get_delta, run_simulation, enrich_frac

_ROOT        = pathlib.Path(__file__).parents[1]
_OUR_DIR     = _ROOT / "rmg_pipeline" / "output" / "isotope" / "iso"
_GOLDMAN_DIR = _ROOT / "rmg_inputs" / "goldmanm-RMG_isotopes_paper_data-234bd52"
_EXP_DATA    = _GOLDMAN_DIR / "exp_data"
_OUT_DIR     = _ROOT / "rmg_pipeline" / "output"

TEMPERATURES = [800, 850, 900, 950]   # °C
PSIA_VALUES  = np.linspace(-10, 20, 5)

RELATIONSHIPS = [
    ("dC2H4 = f(dCH4)",  "methane", "ethene"),
    ("dC2H6 = f(dCH4)",  "methane", "ethane"),
    ("dC2H6 = f(dC2H4)", "ethene",  "ethane"),
    ("dBulk = f(dCH4)",  "methane", "bulk"),
]

GOLDMAN_CLUSTERS = {"propane": 26, "methane": 19, "ethene": 22, "ethane": 18}
OUR_CLUSTERS     = {"propane": 26, "methane": 19, "ethene": 18, "ethane": 25}

OUR_SPECIES_MAP: dict[str, tuple[int, int, int]] = {
    "C(31)":         (19, 0, 1),
    "C(32)":         (19, 1, 1),
    "C2H4(33)":      (18, 0, 2),
    "C2H4(34)":      (18, 2, 2),
    "C2H4(35)":      (18, 1, 2),
    "CC(7)":         (25, 0, 2),
    "CC(8)":         (25, 2, 2),
    "CC(9)":         (25, 1, 2),
    "propane_ooo(1)":(26, 0, 3),
    "CCC(2)":        (26, 3, 3),
    "CCC(3)":        (26, 2, 3),
    "CCC(4)":        (26, 2, 3),
    "CCC(5)":        (26, 1, 3),
    "CCC(6)":        (26, 1, 3),
}

T_COLORS = {800: "#1f77b4", 850: "#ff7f0e", 900: "#2ca02c", 950: "#d62728"}
T_MARKERS = {800: "o", 850: "s", 900: "^", 950: "D"}


# ── δ¹³C helpers ──────────────────────────────────────────────────────────────

def get_delta_ours(concentrations, cluster_num: int, ref_ratio: float = VPDB_RATIO) -> float:
    n = d = 0.0
    for sp, (cn, n13, tc) in OUR_SPECIES_MAP.items():
        if cn != cluster_num:
            continue
        c = concentrations[sp] if isinstance(concentrations, dict) else float(concentrations.get(sp, 0.0))
        n += c * n13
        d += c * (tc - n13)
    if d <= 0 or np.isclose(d, 0.0):
        return np.nan
    return (n / d / ref_ratio - 1.0) * 1000.0


def raw_ratio_ours(concentrations, cluster_num: int) -> float:
    """Return raw ¹³C/¹²C ratio (no reference normalisation)."""
    n = d = 0.0
    for sp, (cn, n13, tc) in OUR_SPECIES_MAP.items():
        if cn != cluster_num:
            continue
        c = concentrations[sp] if isinstance(concentrations, dict) else float(concentrations.get(sp, 0.0))
        n += c * n13
        d += c * (tc - n13)
    if d <= 0:
        return np.nan
    return n / d


# ── init_x builders ───────────────────────────────────────────────────────────

def our_init_x(psia: float, f: float = 0.0049, center_delta: float = -28.0) -> dict:
    e = enrich_frac(center_delta + psia)
    c = enrich_frac(center_delta)
    return {
        "propane_ooo(1)": f * (1 - c) * (1 - e) ** 2,
        "CCC(5)":         f * 2 * e * (1 - c) * (1 - e),
        "CCC(6)":         f * c * (1 - e) ** 2,
        "CCC(3)":         f * (1 - c) * e ** 2,
        "CCC(4)":         f * 2 * e * (1 - e) * c,
        "CCC(2)":         f * c * e ** 2,
        "He":             1.0 - f,
    }


def goldman_init_x(psia: float, f: float = 0.0049, center_delta: float = -28.0) -> dict:
    e = enrich_frac(center_delta + psia)
    c = enrich_frac(center_delta)
    return {
        "CCC":   f * (1 - c) * (1 - e) ** 2,
        "CCC-5": f * 2 * e * (1 - c) * (1 - e),
        "CCC-6": f * c * (1 - e) ** 2,
        "CCC-3": f * (1 - c) * e ** 2,
        "CCC-4": f * 2 * e * (1 - e) * c,
        "CCC-2": f * c * e ** 2,
        "[He]":  1.0 - f,
    }


# ── sweep runners ─────────────────────────────────────────────────────────────

def sweep_ours(yaml_path: pathlib.Path) -> dict[int, pd.DataFrame]:
    """Return {T_C: DataFrame(index=psia, columns=molecules)} for our model."""
    results = {}
    for T_C in TEMPERATURES:
        T_K, t_final = T_C + 273.15, 95.0 / T_C
        rows = []
        for psia in PSIA_VALUES:
            gas    = ct.Solution(str(yaml_path))
            init_x = our_init_x(psia)
            df     = run_simulation(gas, [t_final], T_K=T_K, P_Pa=2e5, init_x=init_x)
            final  = df.iloc[0]
            row = {mol: get_delta_ours(final, OUR_CLUSTERS[mol])
                   for mol in ("methane", "ethene", "ethane")}
            row["bulk"] = get_delta_ours(init_x, OUR_CLUSTERS["propane"])
            # raw ratios for VPDB check
            row["raw_CH4"] = raw_ratio_ours(final, OUR_CLUSTERS["methane"])
            rows.append(row)
            print(f"  ours T={T_C}°C psia={psia:+5.1f}  "
                  f"δCH4={row['methane']:+7.1f}  raw={row['raw_CH4']:.5f}")
        results[T_C] = pd.DataFrame(rows, index=PSIA_VALUES)
    return results


def sweep_goldman(yaml_path: pathlib.Path, csv_path: pathlib.Path) -> dict[int, pd.DataFrame]:
    """Return {T_C: DataFrame(index=psia, columns=molecules)} for Goldman's model."""
    ci = pd.read_csv(csv_path, index_col="name")
    results = {}
    for T_C in TEMPERATURES:
        T_K, t_final = T_C + 273.15, 95.0 / T_C
        rows = []
        for psia in PSIA_VALUES:
            gas    = ct.Solution(str(yaml_path))
            init_x = goldman_init_x(psia)
            df     = run_simulation(gas, [t_final], T_K=T_K, P_Pa=2e5, init_x=init_x)
            final  = df.iloc[0]
            row = {mol: get_delta(final, ci, GOLDMAN_CLUSTERS[mol])
                   for mol in ("methane", "ethene", "ethane")}
            row["bulk"] = get_delta(pd.Series(init_x), ci, GOLDMAN_CLUSTERS["propane"])
            rows.append(row)
            print(f"  goldman T={T_C}°C psia={psia:+5.1f}  δCH4={row['methane']:+7.1f}")
        results[T_C] = pd.DataFrame(rows, index=PSIA_VALUES)
    return results


# ── VPDB check ────────────────────────────────────────────────────────────────

def print_vpdb_check(ours_data: dict[int, pd.DataFrame]) -> None:
    """Show raw ¹³C/¹²C ratios vs δ to verify reference ratio is not the issue."""
    print("\n" + "="*60)
    print("VPDB reference check (T=850°C, all PSIA values)")
    print("  VPDB_RATIO used =", VPDB_RATIO)
    print(f"  At δ=-28‰ (propane centre): R = {enrich_frac(-28.0) / (1 - enrich_frac(-28.0)):.6f}")
    print(f"  δ=0‰ corresponds to R = {VPDB_RATIO:.6f}")
    print(f"  δ=+265‰ corresponds to R = {(1 + 265/1000)*VPDB_RATIO:.6f}")
    print()
    df = ours_data[850]
    for psia, row in df.iterrows():
        ratio = row.get("raw_CH4", np.nan)
        delta = row["methane"]
        # What VPDB_RATIO would give δ=0 with this raw ratio?
        implied_ref = ratio if not np.isnan(ratio) else np.nan
        print(f"  psia={psia:+5.1f}  raw_R={ratio:.6f}  δCH4={delta:+7.1f}‰  "
              f"  (need R_ref={ratio:.6f} to give δ=0; VPDB={VPDB_RATIO:.6f})")
    print("="*60)


# ── plotting ──────────────────────────────────────────────────────────────────

def make_figure6(
    ours_data: dict[int, pd.DataFrame],
    goldman_data: dict[int, pd.DataFrame],
    gilbert_values: pd.DataFrame,
    gilbert_unc: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (label, x_mol, y_mol) in zip(axes, RELATIONSHIPS):
        for T_C in TEMPERATURES:
            color  = T_COLORS[T_C]
            marker = T_MARKERS[T_C]

            # ── Our model ────────────────────────────────────────────────────
            our_df = ours_data[T_C]
            ox = our_df[x_mol].values
            oy = our_df[y_mol].values
            mask = np.isfinite(ox) & np.isfinite(oy)
            if mask.sum() >= 2:
                ax.scatter(ox[mask], oy[mask], c=color, marker=marker,
                           s=50, zorder=3, label=f"Ours {T_C}°C" if label == RELATIONSHIPS[0][0] else "")
                sl, ic, *_ = linregress(ox[mask], oy[mask])
                xl = np.array([ox[mask].min(), ox[mask].max()])
                ax.plot(xl, sl * xl + ic, color=color, lw=1.5, ls="-")

            # ── Goldman model ─────────────────────────────────────────────────
            gdf = goldman_data[T_C]
            gx = gdf[x_mol].values
            gy = gdf[y_mol].values
            mask = np.isfinite(gx) & np.isfinite(gy)
            if mask.sum() >= 2:
                ax.scatter(gx[mask], gy[mask], c=color, marker=marker,
                           s=50, facecolors="none", zorder=3,
                           label=f"Goldman {T_C}°C" if label == RELATIONSHIPS[0][0] else "")
                sl, ic, *_ = linregress(gx[mask], gy[mask])
                xl = np.array([gx[mask].min(), gx[mask].max()])
                ax.plot(xl, sl * xl + ic, color=color, lw=1.5, ls="--")

        # ── Gilbert experimental slopes ───────────────────────────────────────
        # Draw each temperature's experimental slope as a band through the
        # midpoint of Goldman's data range (reference x=0, y=0 anchor).
        for T_C in TEMPERATURES:
            color = T_COLORS[T_C]
            exp_sl  = gilbert_values.loc[label, T_C]
            exp_unc = gilbert_unc.loc[label, T_C]
            gdf = goldman_data[T_C]
            gx  = gdf[x_mol].dropna().values
            if len(gx) == 0:
                continue
            x_mid = gx.mean()
            y_mid = (goldman_data[T_C][y_mol].dropna().values).mean()
            xl = np.array([gx.min(), gx.max()])
            # slope lines through centroid of Goldman data
            ax.fill_between(xl,
                            (exp_sl - exp_unc) * (xl - x_mid) + y_mid,
                            (exp_sl + exp_unc) * (xl - x_mid) + y_mid,
                            color=color, alpha=0.12, zorder=1)
            ax.plot(xl,
                    exp_sl * (xl - x_mid) + y_mid,
                    color=color, lw=1, ls=":", zorder=2)

        ax.set_xlabel(f"δ¹³C {x_mol} (‰)")
        ax.set_ylabel(f"δ¹³C {y_mol} (‰)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    # Shared legend
    solid  = plt.Line2D([], [], color="k", ls="-",  lw=1.5, label="Ours (solid fill)")
    dashed = plt.Line2D([], [], color="k", ls="--", lw=1.5, label="Goldman (open)")
    dotted = plt.Line2D([], [], color="k", ls=":",  lw=1,   label="Gilbert exp. (±1σ band)")
    handles = [solid, dashed, dotted]
    for T_C in TEMPERATURES:
        handles.append(plt.Line2D([], [], color=T_COLORS[T_C], marker=T_MARKERS[T_C],
                                  ls="none", ms=7, label=f"{T_C}°C"))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle("Figure 6 replica — δ¹³C slopes (ours vs Goldman vs Gilbert 2016)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ── per-slope breakdown table ─────────────────────────────────────────────────

def print_slope_table(
    ours_data: dict[int, pd.DataFrame],
    goldman_data: dict[int, pd.DataFrame],
    gilbert_values: pd.DataFrame,
    gilbert_unc: pd.DataFrame,
) -> None:
    print("\n" + "="*72)
    print(f"{'Relationship':<24}  {'T':>4}  {'Gilbert':>7}  {'Goldman':>7}  {'Ours':>8}  {'Ours z':>7}  {'GS z':>6}")
    print("="*72)
    for label, x_mol, y_mol in RELATIONSHIPS:
        for T_C in TEMPERATURES:
            exp_sl  = gilbert_values.loc[label, T_C]
            exp_unc = gilbert_unc.loc[label, T_C]

            def _slope(data):
                df = data[T_C]
                x, y = df[x_mol].values, df[y_mol].values
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 2:
                    return np.nan
                return linregress(x[mask], y[mask])[0]

            g_sl = _slope(goldman_data)
            o_sl = _slope(ours_data)
            g_z  = (exp_sl - g_sl) / exp_unc if np.isfinite(g_sl) else np.nan
            o_z  = (exp_sl - o_sl) / exp_unc if np.isfinite(o_sl) else np.nan
            print(f"{label:<24}  {T_C:>4}  {exp_sl:>7.3f}  {g_sl:>7.3f}  {o_sl:>8.3f}  {o_z:>+7.1f}  {g_z:>+6.1f}")
        print()


# ── 850 °C comparison plot ────────────────────────────────────────────────────

def make_figure6_850(
    ours_data:     dict[int, pd.DataFrame],
    goldman_data:  dict[int, pd.DataFrame],
    gilbert_values: pd.DataFrame,
    gilbert_unc:    pd.DataFrame,
    T_C: int = 850,
) -> plt.Figure:
    """4-panel scatter plot at a single temperature comparing three sources.

    Each panel shows:
      ● Our model          — filled circles, solid fit line
      ○ Goldman full model — open circles, dashed fit line
      ─ Gilbert experiment — dotted slope line anchored at data centroid,
                             ±1σ shaded band
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    our_df     = ours_data[T_C]
    goldman_df = goldman_data[T_C]
    color_ours    = "#e05c2a"   # orange-red
    color_goldman = "#2a6bb5"   # blue
    color_gilbert = "#2d8f3c"   # green

    for ax, (label, x_mol, y_mol) in zip(axes, RELATIONSHIPS):
        # ── our model ────────────────────────────────────────────────────────
        ox = our_df[x_mol].values
        oy = our_df[y_mol].values
        mask_o = np.isfinite(ox) & np.isfinite(oy)
        ax.scatter(ox[mask_o], oy[mask_o], color=color_ours, marker="o",
                   s=60, zorder=4, label="Ours (850 °C)")
        if mask_o.sum() >= 2:
            sl, ic, *_ = linregress(ox[mask_o], oy[mask_o])
            xl = np.array([ox[mask_o].min(), ox[mask_o].max()])
            ax.plot(xl, sl * xl + ic, color=color_ours, lw=2, ls="-")

        # ── Goldman model ─────────────────────────────────────────────────────
        gx = goldman_df[x_mol].values
        gy = goldman_df[y_mol].values
        mask_g = np.isfinite(gx) & np.isfinite(gy)
        ax.scatter(gx[mask_g], gy[mask_g], color=color_goldman, marker="o",
                   s=60, zorder=3, facecolors="none", linewidths=1.5,
                   label="Goldman (850 °C)")
        if mask_g.sum() >= 2:
            sl, ic, *_ = linregress(gx[mask_g], gy[mask_g])
            xl = np.array([gx[mask_g].min(), gx[mask_g].max()])
            ax.plot(xl, sl * xl + ic, color=color_goldman, lw=2, ls="--")

        # ── Gilbert experiment ────────────────────────────────────────────────
        exp_sl  = gilbert_values.loc[label, T_C]
        exp_unc = gilbert_unc.loc[label, T_C]
        # Anchor slope line at the centroid of the Goldman scatter data
        x_anchor = float(goldman_df[x_mol].dropna().mean())
        y_anchor = float(goldman_df[y_mol].dropna().mean())
        # x range: span the full range of both models
        all_x = np.concatenate([ox[mask_o], gx[mask_g]])
        xl = np.linspace(all_x.min(), all_x.max(), 100)
        ax.fill_between(xl,
                        (exp_sl - exp_unc) * (xl - x_anchor) + y_anchor,
                        (exp_sl + exp_unc) * (xl - x_anchor) + y_anchor,
                        color=color_gilbert, alpha=0.18, zorder=1)
        ax.plot(xl, exp_sl * (xl - x_anchor) + y_anchor,
                color=color_gilbert, lw=2, ls=":", label="Gilbert 2016 (±1σ)")

        ax.set_xlabel(f"δ¹³C {x_mol} (‰)", fontsize=10)
        ax.set_ylabel(f"δ¹³C {y_mol} (‰)", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        if label == RELATIONSHIPS[0][0]:
            ax.legend(fontsize=9, framealpha=0.9)

    fig.suptitle(f"δ¹³C slopes at {T_C} °C — Gilbert 2016 vs Goldman vs Ours",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    gilbert_values = pd.read_csv(_EXP_DATA / "Gilbert_table2_values.csv",     index_col="relationship")
    gilbert_unc    = pd.read_csv(_EXP_DATA / "Gilbert_table2_uncertainty.csv", index_col="relationship")
    gilbert_values.columns = [int(c) for c in gilbert_values.columns]
    gilbert_unc.columns    = [int(c) for c in gilbert_unc.columns]

    print("\nRunning ours...")
    t0 = time.perf_counter()
    ours_data = sweep_ours(_OUR_DIR / "chem.yaml")
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    print("\nRunning Goldman full model...")
    t0 = time.perf_counter()
    goldman_data = sweep_goldman(
        _GOLDMAN_DIR / "mechanisms" / "full_model" / "chem.yaml",
        _GOLDMAN_DIR / "mechanisms" / "full_model" / "isotopomer_cluster_info.csv",
    )
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    print_vpdb_check(ours_data)
    print_slope_table(ours_data, goldman_data, gilbert_values, gilbert_unc)

    fig = make_figure6(ours_data, goldman_data, gilbert_values, gilbert_unc)
    out = _OUT_DIR / "figure6_replica.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")

    fig850 = make_figure6_850(ours_data, goldman_data, gilbert_values, gilbert_unc)
    out850 = _OUT_DIR / "figure6_850.png"
    fig850.savefig(out850, dpi=150, bbox_inches="tight")
    print(f"Saved: {out850}")


if __name__ == "__main__":
    main()
