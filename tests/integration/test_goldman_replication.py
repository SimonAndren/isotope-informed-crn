"""Goldman 2019 Table 2 replication tests.

Validates that the delta-13C slope analysis pipeline produces correct results
when run against Goldman's pre-generated mechanism YAMLs. These tests serve as
regression guards for the analysis code and as baseline comparisons for our
generated mechanism.

The full-model test (Table 2 = 0.5) confirms the analysis pipeline is correct.
The slope-range tests ensure physically meaningful values across temperatures.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
import cantera as ct
from scipy.stats import linregress

from isotopologue.goldman import VPDB_RATIO, get_delta, run_simulation, enrich_frac

_ROOT = pathlib.Path(__file__).parents[2]
_GOLDMAN_DIR = _ROOT / "rmg_inputs" / "goldmanm-RMG_isotopes_paper_data-234bd52"
_EXP_DATA = _GOLDMAN_DIR / "exp_data"
_FULL_MODEL = _GOLDMAN_DIR / "mechanisms" / "full_model"
_THREE_RXN = _GOLDMAN_DIR / "mechanisms" / "three_reaction_model"

TEMPERATURES = [800, 850, 900, 950]
PSIA_VALUES = np.linspace(-10, 20, 5)

RELATIONSHIPS = [
    ("dC2H4 = f(dCH4)", "methane", "ethene"),
    ("dC2H6 = f(dCH4)", "methane", "ethane"),
    ("dC2H6 = f(dC2H4)", "ethene", "ethane"),
    ("dBulk = f(dCH4)", "methane", "bulk"),
]

GOLDMAN_CLUSTERS_FULL = {"propane": 26, "methane": 19, "ethene": 22, "ethane": 18}
GOLDMAN_CLUSTERS_3RXN = {"propane": 5, "methane": 0, "ethene": 1, "ethane": 4}


def _goldman_init_x(psia: float, f: float = 0.0049, center_delta: float = -28.0) -> dict:
    e = enrich_frac(center_delta + psia)
    c = enrich_frac(center_delta)
    return {
        "CCC": f * (1 - c) * (1 - e) ** 2,
        "CCC-5": f * 2 * e * (1 - c) * (1 - e),
        "CCC-6": f * c * (1 - e) ** 2,
        "CCC-3": f * (1 - c) * e ** 2,
        "CCC-4": f * 2 * e * (1 - e) * c,
        "CCC-2": f * c * e ** 2,
        "[He]": 1.0 - f,
    }


def _compute_slopes(yaml_path, csv_path, clusters, init_x_fn):
    """PSIA x temperature sweep -> slope DataFrame."""
    ci = pd.read_csv(csv_path, index_col="name")
    slopes = pd.DataFrame(
        index=[r[0] for r in RELATIONSHIPS], columns=TEMPERATURES, dtype=float,
    )
    for T_C in TEMPERATURES:
        T_K, t_final = T_C + 273.15, 95.0 / T_C
        rows = []
        for psia in PSIA_VALUES:
            gas = ct.Solution(str(yaml_path))
            init_x = init_x_fn(psia=psia)
            df = run_simulation(gas, [t_final], T_K=T_K, P_Pa=2e5, init_x=init_x)
            final = df.iloc[0]
            row = {
                mol: get_delta(final, ci, cluster_num=clusters[mol])
                for mol in ("methane", "ethene", "ethane")
            }
            row["bulk"] = get_delta(pd.Series(init_x), ci, cluster_num=clusters["propane"])
            rows.append(row)
        enrich = pd.DataFrame(rows, index=PSIA_VALUES)
        for label, x_mol, y_mol in RELATIONSHIPS:
            x = enrich[x_mol].values
            y = enrich[y_mol].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2:
                slope, *_ = linregress(x[mask], y[mask])
                slopes.loc[label, T_C] = slope
    return slopes


def _table2_score(slopes, gilbert_values, gilbert_unc):
    scaled = (gilbert_values - slopes) / gilbert_unc
    vals = scaled.values[np.isfinite(scaled.values)]
    return np.std(vals) if len(vals) > 0 else float("nan")


@pytest.fixture(scope="module")
def gilbert_data():
    values = pd.read_csv(_EXP_DATA / "Gilbert_table2_values.csv", index_col="relationship")
    unc = pd.read_csv(_EXP_DATA / "Gilbert_table2_uncertainty.csv", index_col="relationship")
    values.columns = [int(c) for c in values.columns]
    unc.columns = [int(c) for c in unc.columns]
    return values, unc


# ── Goldman full-model replication ───────────────────────────────────────────


@pytest.mark.slow
class TestGoldmanFullModelTable2:
    """Goldman's full model must reproduce Table 2 score ~ 0.5.

    This is the ground-truth validation: if our analysis pipeline gives the
    correct score on Goldman's pre-generated mechanism, the pipeline is correct.
    """

    @pytest.fixture(scope="class")
    def full_slopes(self):
        return _compute_slopes(
            _FULL_MODEL / "chem.yaml",
            _FULL_MODEL / "isotopomer_cluster_info.csv",
            GOLDMAN_CLUSTERS_FULL,
            _goldman_init_x,
        )

    def test_table2_score_within_tolerance(self, full_slopes, gilbert_data):
        """Goldman full model Table 2 score must be < 1.0 (published: 0.5)."""
        values, unc = gilbert_data
        score = _table2_score(full_slopes, values, unc)
        assert score < 1.0, f"Goldman full model Table 2 = {score:.2f}, expected < 1.0"

    def test_dC2H6_slope_near_unity(self, full_slopes):
        """dC2H6 = f(dCH4) slopes must be near 1.0 (0.97-1.0 range)."""
        for T_C in TEMPERATURES:
            slope = full_slopes.loc["dC2H6 = f(dCH4)", T_C]
            assert 0.95 < slope < 1.05, (
                f"dC2H6 slope at {T_C}C = {slope:.3f}, expected ~1.0"
            )

    def test_dC2H4_slope_below_unity(self, full_slopes):
        """dC2H4 = f(dCH4) slopes must be < 1.0 (ethylene fractionates less)."""
        for T_C in TEMPERATURES:
            slope = full_slopes.loc["dC2H4 = f(dCH4)", T_C]
            assert 0.3 < slope < 0.8, (
                f"dC2H4 slope at {T_C}C = {slope:.3f}, expected 0.3-0.8"
            )

    def test_dBulk_slope_near_two_thirds(self, full_slopes):
        """dBulk = f(dCH4) slopes must be near 2/3 (mass balance constraint)."""
        for T_C in TEMPERATURES:
            slope = full_slopes.loc["dBulk = f(dCH4)", T_C]
            assert 0.5 < slope < 0.8, (
                f"dBulk slope at {T_C}C = {slope:.3f}, expected ~0.667"
            )

    def test_slopes_increase_with_temperature(self, full_slopes):
        """dC2H4 slopes must increase with T (less fractionation at higher T)."""
        dC2H4_slopes = [full_slopes.loc["dC2H4 = f(dCH4)", T] for T in TEMPERATURES]
        for i in range(len(dC2H4_slopes) - 1):
            assert dC2H4_slopes[i] < dC2H4_slopes[i + 1], (
                f"dC2H4 slope should increase with T: "
                f"{TEMPERATURES[i]}C={dC2H4_slopes[i]:.3f} >= "
                f"{TEMPERATURES[i+1]}C={dC2H4_slopes[i+1]:.3f}"
            )


# ── Single-temperature quick validation ──────────────────────────────────────


class TestGoldman850CDelta:
    """Quick delta-13C checks at 850C on the 3-rxn model (fast, no full sweep)."""

    @pytest.fixture(scope="class")
    def sim_850(self):
        gas = ct.Solution(str(_THREE_RXN / "chem.yaml"))
        ci = pd.read_csv(
            _THREE_RXN / "isotopomer_cluster_info.csv", index_col="name"
        )
        init_x = _goldman_init_x(psia=0.0)
        T_K = 850 + 273.15
        df = run_simulation(gas, [95.0 / T_K], T_K=T_K, P_Pa=2e5, init_x=init_x)
        return df.iloc[0], ci

    def test_methane_delta_negative(self, sim_850):
        """delta-CH4 must be negative (lighter than VPDB) for propane pyrolysis."""
        final, ci = sim_850
        d = get_delta(final, ci, cluster_num=GOLDMAN_CLUSTERS_3RXN["methane"])
        assert np.isfinite(d)
        assert d < 0, f"delta-CH4 = {d:+.1f}, expected negative"

    def test_methane_delta_in_range(self, sim_850):
        """delta-CH4 must be in [-100, 0] permil range."""
        final, ci = sim_850
        d = get_delta(final, ci, cluster_num=GOLDMAN_CLUSTERS_3RXN["methane"])
        assert -100 < d < 0, f"delta-CH4 = {d:+.1f}, expected (-100, 0)"

    def test_ethylene_delta_finite(self, sim_850):
        final, ci = sim_850
        d = get_delta(final, ci, cluster_num=GOLDMAN_CLUSTERS_3RXN["ethene"])
        assert np.isfinite(d)
        assert -200 < d < 200

    def test_ethane_delta_finite(self, sim_850):
        final, ci = sim_850
        d = get_delta(final, ci, cluster_num=GOLDMAN_CLUSTERS_3RXN["ethane"])
        assert np.isfinite(d)
        assert -200 < d < 200
