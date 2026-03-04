"""Goldman 2019 δ¹³C analysis for isotopologue-expanded Cantera mechanisms.

Ports Goldman's analysis_methods.py and cantera_tools.py to work with:
  - Cantera YAML format (instead of deprecated .cti)
  - pandas 2.x  (pd.concat, .items()  — not DataFrame.append / .iteritems())
  - Python 3    (r.thermo  — not reactor.kinetics)

Reference ratio matches Goldman 2019 / Gilbert 2016:
    0.011115  (NGS-2 Propane, Hut et al. 1987)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cantera as ct

# Reference ¹³C/¹²C ratio used throughout Goldman 2019
VPDB_RATIO = 0.011115  # NGS-2 Propane, Hut et al. 1987


def enrich_frac(delta: float, ref_ratio: float = VPDB_RATIO) -> float:
    """Convert δ¹³C (‰) to isotope fraction.

    Inverse of: delta = (ratio / ref_ratio - 1) * 1000
    """
    ratio = (delta / 1000.0 + 1.0) * ref_ratio
    return ratio / (1.0 + ratio)


def propane_init_x(
    fraction_propane: float = 0.0049,
    delta_total: float = -28.0,
    psia: float = 5.4,
) -> dict[str, float]:
    """Goldman 2019 position-specific propane isotopologue mole fractions.

    Propane = CH₃–CH₂–CH₃ (edge–center–edge, C1–C2–C3).

        edge_delta   = delta_total + psia / 3
        center_delta = delta_total − 2 * psia / 3

    The six SMILES-labelled isotopologues:
        CCC   = 0 ¹³C          CCC-2 = all 3 ¹³C
        CCC-3 = both edges ¹³C CCC-4 = one edge + center ¹³C
        CCC-5 = one edge ¹³C   CCC-6 = center only ¹³C

    Returns a dict keyed by Cantera species name (CCC, CCC-2…CCC-6, [He]).
    """
    e = enrich_frac(delta_total + psia / 3.0)        # edge fraction
    c = enrich_frac(delta_total - 2.0 * psia / 3.0)  # center fraction
    f = fraction_propane
    return _propane_isotopologue_x(f, e, c)


def table2_init_x(
    psia: float,
    fraction_propane: float = 0.0049,
    center_delta: float = -28.0,
) -> dict[str, float]:
    """Goldman 2019 Table 2 / Figure 6 propane initial conditions.

    Center carbon δ¹³C is fixed at center_delta; edge δ¹³C = center_delta + psia.

    This differs from propane_init_x (which holds bulk δ constant): here only
    the edge varies, so the bulk δ¹³C shifts as psia changes:
        bulk = center_delta + 2 * psia / 3

    Use this for Table 2 slope analysis; use propane_init_x for Figures 2–3.
    """
    e = enrich_frac(center_delta + psia)  # edge varies
    c = enrich_frac(center_delta)          # center fixed
    return _propane_isotopologue_x(fraction_propane, e, c)


def _propane_isotopologue_x(f: float, e: float, c: float) -> dict[str, float]:
    """Build propane isotopologue mole fractions from edge (e) and center (c) fractions."""
    return {
        "CCC":   f * (1 - c) * (1 - e) ** 2,
        "CCC-2": f * c * e ** 2,
        "CCC-3": f * (1 - c) * e ** 2,
        "CCC-4": f * 2 * e * (1 - e) * c,
        "CCC-5": f * 2 * e * (1 - c) * (1 - e),
        "CCC-6": f * c * (1 - e) ** 2,
        "[He]":  1.0 - f,
    }


def get_delta(
    concentrations: pd.Series,
    cluster_info: pd.DataFrame,
    cluster_num: int,
    ref_ratio: float = VPDB_RATIO,
) -> float:
    """δ¹³C (‰) for a cluster of isotopologues.

    Args:
        concentrations: pd.Series of concentrations (or mole fractions) indexed
            by species name. Units cancel in the ratio, so either works.
        cluster_info:   DataFrame from isotopomer_cluster_info.csv (index = name).
        cluster_num:    Cluster number to compute δ for.
        ref_ratio:      Reference ¹³C/¹²C ratio.

    Returns:
        δ¹³C in ‰, or nan if the cluster concentration is effectively zero.
    """
    labels = cluster_info.index[cluster_info["cluster_number"] == cluster_num]
    conc = concentrations.reindex(labels, fill_value=0.0)
    denom = (conc * cluster_info.loc[labels, "unenriched_atoms"]).sum()
    if denom <= 0 or np.isclose(denom, 0.0, atol=conc.sum() * 1e-4):
        return np.nan
    numer = (conc * cluster_info.loc[labels, "enriched_atoms"]).sum()
    return (numer / denom / ref_ratio - 1.0) * 1000.0


def get_psie(
    concentrations: pd.Series,
    cluster_info: pd.DataFrame,
    cluster_num: int,
    type1: str,
    type2: str,
    ref_ratio: float = VPDB_RATIO,
) -> float:
    """Position-specific isotope enrichment: δ(type1) − δ(type2).

    Args:
        type1, type2: Column prefixes in cluster_info (e.g. "1", "2", "r", "not_r").
            The function reads "{type}_enriched" and "{type}_unenriched" columns.
    """
    labels = cluster_info.index[cluster_info["cluster_number"] == cluster_num]
    conc = concentrations.reindex(labels, fill_value=0.0)

    def _delta(col_e: str, col_u: str) -> float:
        denom = (conc * cluster_info.loc[labels, col_u]).sum()
        if denom <= 0 or np.isclose(denom, 0.0, atol=conc.sum() * 1e-4):
            return np.nan
        numer = (conc * cluster_info.loc[labels, col_e]).sum()
        return (numer / denom / ref_ratio - 1.0) * 1000.0

    d1 = _delta(f"{type1}_enriched", f"{type1}_unenriched")
    d2 = _delta(f"{type2}_enriched", f"{type2}_unenriched")
    if np.isnan(d1) or np.isnan(d2):
        return np.nan
    return d1 - d2


def run_simulation(
    gas: ct.Solution,
    times: np.ndarray,
    T_K: float,
    P_Pa: float,
    init_x: dict[str, float],
    atol: float = 1e-15,
    rtol: float = 1e-9,
) -> pd.DataFrame:
    """Constant-T-and-P Cantera simulation.

    Args:
        gas:     Cantera Solution object (mechanism).
        times:   Array of output times (s). The reactor advances to each in order.
        T_K:     Temperature (K).
        P_Pa:    Pressure (Pa).
        init_x:  Initial mole fractions {species_name: mole_fraction}.
        atol/rtol: ODE solver tolerances.

    Returns:
        DataFrame of molar concentrations (kmol/m³), indexed by time (s),
        columns = species names in mechanism order.
    """
    gas.TPX = T_K, P_Pa, init_x
    r = ct.IdealGasConstPressureReactor(gas, energy="off", clone=False)
    sim = ct.ReactorNet([r])
    sim.atol = atol
    sim.rtol = rtol

    species_names = r.phase.species_names
    rows = []
    for t in times:
        sim.advance(t)
        density = r.phase.density_mole
        rows.append(pd.Series(r.phase.X * density, index=species_names, name=t))
    return pd.DataFrame(rows)
