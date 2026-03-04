"""Integration tests comparing engine output to Gilbert 2016 experimental data.

Gilbert et al. (2016) measured position-specific ¹³C enrichment in propane
pyrolysis products at 800–950 °C. Goldman et al. (2019) summarise the key
observable as enrichment slopes: δ¹³C(product) = slope × δ¹³C(CH4) + intercept.
The slopes from Gilbert Table 2 are the regression target.

Data files: rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52/exp_data/
    Gilbert_table2_values.csv    — measured slopes at 4 temperatures
    Gilbert_table2_uncertainty.csv — 1σ uncertainties on those slopes

Current engine limitations
---------------------------
The propane benchmark in `benchmarks/propane.py` does not yet have:
  1. Correct atom maps for exchange reactions (carbon routing is wrong).
  2. Site-specific KIEs.

Because of (1) the engine will not correctly reproduce enrichment slopes.
Tests marked `xfail` document the expected failure and will auto-pass once
atom maps + KIEs are implemented (see TODO.md).

Tests that do NOT require correct isotope fractionation (structure, mass
balance, temperature scaling) are expected to pass immediately.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from isotopologue.analysis import delta, site_delta, total_ratio_vectorized
from isotopologue.benchmarks.propane import initial_conditions, propane_3rxn, propane_6rxn
from isotopologue.engine import IsotopologueEngine

# ─── Paths ────────────────────────────────────────────────────────────────────

_EXP_DIR = pathlib.Path(__file__).parents[2] / (
    "rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52/exp_data"
)
_TABLE2_VALUES = _EXP_DIR / "Gilbert_table2_values.csv"
_TABLE2_UNCERT = _EXP_DIR / "Gilbert_table2_uncertainty.csv"

# Temperatures in K matching the paper (800, 850, 900, 950 °C)
TEMPERATURES_K = [1073.0, 1123.0, 1173.0, 1223.0]
TEMPERATURES_C = [800, 850, 900, 950]

# Residence time (Goldman uses ~85 ms)
T_END = 0.085  # seconds


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _run_at(T_K: float, model: str = "3rxn") -> dict[str, np.ndarray]:
    """Run propane pyrolysis at temperature T_K, return final concentrations."""
    net = propane_3rxn(T_K) if model == "3rxn" else propane_6rxn(T_K)
    eng = IsotopologueEngine(net)
    conc = initial_conditions(net)
    y0 = net.pack(conc)
    result = eng.solve(y0, (0, T_END), method="BDF")
    assert result.success, f"Solver failed at {T_K} K: {result.message}"
    return net.unpack(result.y[:, -1])


def _bulk_delta(conc: np.ndarray, n_atoms: int) -> float:
    """Bulk δ¹³C (‰ VPDB) of a species."""
    return delta(total_ratio_vectorized(conc, n_atoms))


def _enrichment_slope(
    d_product: list[float], d_ch4: list[float]
) -> float:
    """Linear regression slope of δ¹³C(product) on δ¹³C(CH4)."""
    x = np.array(d_ch4)
    y = np.array(d_product)
    if x.std() < 1e-6:
        return float("nan")
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])


# ─── Structural tests (no fractionation assumptions) ─────────────────────────


class TestGilbertStructure:
    """These tests check that the engine runs correctly at all 4 temperatures
    and that basic conservation laws hold — independent of KIEs or atom maps."""

    @pytest.mark.slow
    @pytest.mark.parametrize("T_K", TEMPERATURES_K)
    def test_simulation_completes(self, T_K):
        """Engine must complete without error at each experimental temperature."""
        final = _run_at(T_K)
        assert all(c.sum() >= -1e-10 for c in final.values())

    @pytest.mark.slow
    @pytest.mark.parametrize("T_K", TEMPERATURES_K)
    def test_carbon_conservation(self, T_K):
        """Weighted total carbon must be conserved throughout the simulation."""
        n_carbons = {"C3H8": 3, "CH3": 1, "C2H5": 2, "CH4": 1, "C2H4": 2, "C2H6": 2}
        net = propane_3rxn(T_K)
        eng = IsotopologueEngine(net)
        conc = initial_conditions(net)
        y0 = net.pack(conc)

        def total_carbon(y):
            state = net.unpack(y)
            return sum(n_carbons[name] * state[name].sum() for name in net.species)

        initial_c = total_carbon(y0)
        result = eng.solve(
            y0, (0, T_END), method="BDF",
            t_eval=np.linspace(0, T_END, 20)
        )
        assert result.success
        for yi in result.y.T:
            np.testing.assert_allclose(
                total_carbon(yi), initial_c, rtol=1e-3,
                err_msg=f"Carbon not conserved at {T_K:.0f} K"
            )

    @pytest.mark.slow
    def test_higher_temperature_gives_more_conversion(self):
        """Higher temperature should produce more propane cracking products."""
        results = {T: _run_at(T) for T in TEMPERATURES_K}
        propane_remaining = [results[T]["C3H8"].sum() for T in TEMPERATURES_K]
        # Strictly monotone decrease in propane (more cracking at higher T)
        for i in range(len(propane_remaining) - 1):
            assert propane_remaining[i] >= propane_remaining[i + 1] - 1e-10, (
                f"Expected more propane cracking at higher T: "
                f"{TEMPERATURES_C[i]}°C has {propane_remaining[i]:.4f} "
                f"vs {TEMPERATURES_C[i+1]}°C has {propane_remaining[i+1]:.4f}"
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("T_K", TEMPERATURES_K)
    def test_c13_mass_balance(self, T_K):
        """Total ¹³C moles (sum of conc[i] × popcount(i)) must be conserved."""
        net = propane_3rxn(T_K)
        eng = IsotopologueEngine(net)
        conc = initial_conditions(net)
        y0 = net.pack(conc)

        def c13_moles(y):
            state = net.unpack(y)
            total = 0.0
            for name in net.species:
                c = state[name]
                for im in range(len(c)):
                    total += c[im] * bin(im).count("1")
            return total

        initial = c13_moles(y0)
        result = eng.solve(y0, (0, T_END), method="BDF",
                           t_eval=np.linspace(0, T_END, 10))
        assert result.success
        for yi in result.y.T:
            np.testing.assert_allclose(
                c13_moles(yi), initial, rtol=1e-3,
                err_msg=f"¹³C mass balance violated at {T_K:.0f} K"
            )


# ─── Gilbert Table 2 comparison ───────────────────────────────────────────────


@pytest.mark.slow
class TestGilbertTable2:
    """Compare computed enrichment slopes to Gilbert (2016) Table 2.

    Tests here are xfail until atom maps and KIEs are implemented.
    See TODO.md → "Make Gilbert replication test pass".
    """

    @pytest.fixture(scope="class")
    def gilbert_slopes(self):
        """Load Gilbert Table 2 values and uncertainties."""
        values = pd.read_csv(_TABLE2_VALUES, index_col=0)
        uncert = pd.read_csv(_TABLE2_UNCERT, index_col=0)
        return values, uncert

    @pytest.fixture(scope="class")
    def computed_slopes(self):
        """Run engine at all 4 temperatures and compute enrichment slopes.

        Strategy: simulate at natural abundance conditions, perturb initial
        propane δ¹³C by ±20‰, and compute the regression slope of each
        product's δ¹³C vs CH4's δ¹³C across the 4 temperatures.
        """
        d_ch4, d_c2h4, d_c2h6 = [], [], []

        for T_K in TEMPERATURES_K:
            final = _run_at(T_K, model="3rxn")

            ch4_conc = final["CH4"]
            c2h4_conc = final["C2H4"]
            c2h6_conc = final["C2H6"]

            # Only compute δ if there is product (avoid division by zero)
            if ch4_conc.sum() > 1e-15:
                d_ch4.append(_bulk_delta(ch4_conc, 1))
                d_c2h4.append(_bulk_delta(c2h4_conc, 2))
                d_c2h6.append(_bulk_delta(c2h6_conc, 2))
            else:
                d_ch4.append(float("nan"))
                d_c2h4.append(float("nan"))
                d_c2h6.append(float("nan"))

        return {
            "d_ch4": d_ch4,
            "d_c2h4": d_c2h4,
            "d_c2h6": d_c2h6,
        }

    def test_experimental_data_loads(self, gilbert_slopes):
        """Gilbert Table 2 CSV must load and contain the expected rows."""
        values, uncert = gilbert_slopes
        assert "dC2H4 = f(dCH4)" in values.index
        assert "dC2H6 = f(dCH4)" in values.index
        assert list(values.columns) == ["800", "850", "900", "950"]

    def test_products_have_nonzero_concentration(self, computed_slopes):
        """At each temperature CH4 must have formed (conversion > 0)."""
        for i, T_C in enumerate(TEMPERATURES_C):
            d = computed_slopes["d_ch4"][i]
            assert not np.isnan(d), f"No CH4 formed at {T_C}°C — propane not cracking"

    @pytest.mark.xfail(
        reason=(
            "Enrichment slopes require correct atom maps for exchange reactions "
            "and site-specific KIEs. Neither is implemented yet. "
            "See TODO.md → 'Make Gilbert replication test pass'."
        ),
        strict=False,
    )
    def test_c2h4_vs_ch4_slope_within_2sigma(self, gilbert_slopes, computed_slopes):
        """δ¹³C(C2H4) vs δ¹³C(CH4) slope should match Gilbert within 2σ.

        Gilbert values: ~0.50–0.57 across 800–950 °C.
        Tolerance: ±2σ from uncertainty CSV.
        """
        values, uncert = gilbert_slopes
        row = "dC2H4 = f(dCH4)"
        computed = _enrichment_slope(
            computed_slopes["d_c2h4"], computed_slopes["d_ch4"]
        )
        gilbert_mean = float(values.loc[row, "850"])
        gilbert_2sigma = 2 * float(uncert.loc[row, "850"])

        assert abs(computed - gilbert_mean) <= gilbert_2sigma, (
            f"C2H4/CH4 slope: computed={computed:.3f}, "
            f"Gilbert={gilbert_mean:.3f} ± {gilbert_2sigma:.3f}"
        )

    @pytest.mark.xfail(
        reason=(
            "Enrichment slopes require correct atom maps and KIEs. "
            "See TODO.md → 'Make Gilbert replication test pass'."
        ),
        strict=False,
    )
    def test_c2h6_vs_ch4_slope_within_2sigma(self, gilbert_slopes, computed_slopes):
        """δ¹³C(C2H6) vs δ¹³C(CH4) slope should match Gilbert within 2σ.

        Gilbert values: ~0.97–0.98 across temperatures.
        """
        values, uncert = gilbert_slopes
        row = "dC2H6 = f(dCH4)"
        computed = _enrichment_slope(
            computed_slopes["d_c2h6"], computed_slopes["d_ch4"]
        )
        gilbert_mean = float(values.loc[row, "850"])
        gilbert_2sigma = 2 * float(uncert.loc[row, "850"])

        assert abs(computed - gilbert_mean) <= gilbert_2sigma, (
            f"C2H6/CH4 slope: computed={computed:.3f}, "
            f"Gilbert={gilbert_mean:.3f} ± {gilbert_2sigma:.3f}"
        )

    @pytest.mark.xfail(
        reason=(
            "Enrichment slopes require correct atom maps and KIEs. "
            "See TODO.md → 'Make Gilbert replication test pass'."
        ),
        strict=False,
    )
    def test_slope_ordering(self, computed_slopes):
        """Enrichment slope ordering from Gilbert: C2H6/CH4 > C2H4/CH4 > 0.

        This ordering is a purely structural prediction (no quantitative KIE
        needed) and should hold once atom maps are correct.
        """
        slope_c2h4 = _enrichment_slope(
            computed_slopes["d_c2h4"], computed_slopes["d_ch4"]
        )
        slope_c2h6 = _enrichment_slope(
            computed_slopes["d_c2h6"], computed_slopes["d_ch4"]
        )

        assert slope_c2h4 > 0, f"C2H4/CH4 slope should be positive, got {slope_c2h4:.3f}"
        assert slope_c2h6 > 0, f"C2H6/CH4 slope should be positive, got {slope_c2h6:.3f}"
        assert slope_c2h6 > slope_c2h4, (
            f"Gilbert: C2H6/CH4 slope ({slope_c2h6:.3f}) should exceed "
            f"C2H4/CH4 slope ({slope_c2h4:.3f})"
        )
