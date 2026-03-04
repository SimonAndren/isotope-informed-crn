"""Tests for Goldman 2019 δ¹³C replication.

Validates that isotopologue.goldman correctly:
  1. Computes initial propane isotopologue mole fractions from δ_total and PSIA.
  2. Derives δ¹³C from species concentrations using isotopomer_cluster_info.csv.
  3. Runs a Cantera simulation and recovers physically meaningful enrichments.

Cluster numbers for the 3-rxn model (from isotopomer_cluster_info.csv):
    0 = methane (C)   1 = ethene (C=C)   2 = methyl ([CH3])
    3 = ethyl (C[CH2]) 4 = ethane (CC)   5 = propane (CCC)
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import cantera as ct

from isotopologue.goldman import (
    VPDB_RATIO,
    enrich_frac,
    get_delta,
    get_psie,
    propane_init_x,
    run_simulation,
)

_MECH_DIR = pathlib.Path(__file__).parents[2] / (
    "rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52/mechanisms"
)
_THREE_RXN = _MECH_DIR / "three_reaction_model"


def _cluster_info(model_dir: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(model_dir / "isotopomer_cluster_info.csv", index_col="name")


# ─── Initial conditions ───────────────────────────────────────────────────────


class TestInitialConditions:
    def test_mole_fractions_sum_to_one(self):
        init_x = propane_init_x()
        assert abs(sum(init_x.values()) - 1.0) < 1e-12

    def test_propane_total_fraction(self):
        init_x = propane_init_x(fraction_propane=0.0049)
        propane_sum = sum(v for k, v in init_x.items() if "[" not in k)
        assert abs(propane_sum - 0.0049) < 1e-12

    def test_all_fractions_non_negative(self):
        for psia in (-10, 0, 5.4, 20):
            init_x = propane_init_x(psia=psia)
            for k, v in init_x.items():
                assert v >= 0, f"Negative mole fraction for {k} at psia={psia}"

    def test_initial_propane_delta_equals_delta_total(self):
        """Bulk δ¹³C of the initial propane mixture must equal delta_total (−28‰)."""
        init_x = propane_init_x(delta_total=-28.0, psia=5.4)
        ci = _cluster_info(_THREE_RXN)
        concs = pd.Series(init_x)
        delta = get_delta(concs, ci, cluster_num=5)
        assert abs(delta - (-28.0)) < 0.1

    def test_initial_propane_delta_varies_with_delta_total(self):
        """Changing delta_total shifts the computed δ¹³C proportionally."""
        ci = _cluster_info(_THREE_RXN)
        for dt in (-40.0, -28.0, -10.0):
            init_x = propane_init_x(delta_total=dt, psia=0.0)
            delta = get_delta(pd.Series(init_x), ci, cluster_num=5)
            assert abs(delta - dt) < 0.1, f"Expected δ≈{dt}, got {delta:.3f}"

    def test_enrich_frac_roundtrip(self):
        """enrich_frac(delta) → ratio → delta must recover the original delta."""
        for delta_val in (-40.0, -28.0, -10.0, 0.0, 10.0):
            frac = enrich_frac(delta_val)
            ratio = frac / (1.0 - frac)
            delta_recovered = (ratio / VPDB_RATIO - 1.0) * 1000.0
            assert abs(delta_recovered - delta_val) < 1e-6


# ─── δ¹³C computation ─────────────────────────────────────────────────────────


class TestDeltaComputation:
    def test_get_delta_returns_nan_for_zero_concentration(self):
        ci = _cluster_info(_THREE_RXN)
        concs = pd.Series(0.0, index=ci.index)
        result = get_delta(concs, ci, cluster_num=0)
        assert np.isnan(result)

    def test_get_delta_at_natural_abundance_is_zero(self):
        """δ¹³C when concentrations reflect natural abundance must be ≈ 0‰."""
        ci = _cluster_info(_THREE_RXN)
        # Methane cluster (0): C (0 enriched, 1 unenriched) and C-2 (1 enriched, 0 unenriched).
        # At natural abundance: fraction C-2 = p, fraction C = 1-p, where p = VPDB/(1+VPDB).
        p = VPDB_RATIO / (1.0 + VPDB_RATIO)
        concs = pd.Series({"C": 1.0 - p, "C-2": p})
        delta = get_delta(concs, ci, cluster_num=0)
        assert abs(delta) < 0.1

    def test_get_psie_zero_when_no_position_preference(self):
        """PSIE must be 0 when psia=0 (all positions equally enriched)."""
        ci = _cluster_info(_THREE_RXN)
        # psia=0 → edge_delta = center_delta = delta_total → uniform enrichment
        init_x = propane_init_x(delta_total=-28.0, psia=0.0)
        concs = pd.Series(init_x)
        psie = get_psie(concs, ci, cluster_num=5, type1="1", type2="2")
        assert abs(psie) < 1e-6

    def test_get_psie_nonzero_when_position_preference_exists(self):
        """PSIE must be non-zero when psia ≠ 0."""
        ci = _cluster_info(_THREE_RXN)
        init_x = propane_init_x(delta_total=-28.0, psia=5.4)
        concs = pd.Series(init_x)
        psie = get_psie(concs, ci, cluster_num=5, type1="1", type2="2")
        assert abs(psie) > 0.1


# ─── Cantera simulation ───────────────────────────────────────────────────────


@pytest.mark.slow
class TestGoldmanSimulation:
    """Run Goldman's 3-rxn simulation at 850 °C and verify δ¹³C output."""

    @pytest.fixture(scope="class")
    def sim_result(self):
        """(gas, species_df, cluster_info) for 3-rxn at 850 °C."""
        gas = ct.Solution(str(_THREE_RXN / "chem.yaml"))
        ci = _cluster_info(_THREE_RXN)
        init_x = propane_init_x()
        T_K = 850.0 + 273.15
        times = np.linspace(1e-4, 95.0 / T_K, 30)
        df = run_simulation(gas, times, T_K=T_K, P_Pa=2e5, init_x=init_x)
        return gas, df, ci

    def test_dataframe_shape(self, sim_result):
        _, df, _ = sim_result
        assert df.shape == (30, 24)  # 30 time points, 24 species in 3-rxn model

    def test_propane_cluster_decreases(self, sim_result):
        """Total propane (summed over all isotopologues) must decrease."""
        _, df, ci = sim_result
        labels = ci.index[ci["cluster_number"] == 5]
        assert df.iloc[-1][labels].sum() < df.iloc[0][labels].sum()

    def test_product_clusters_increase(self, sim_result):
        """Methane and ethene clusters must grow from near-zero."""
        _, df, ci = sim_result
        for cluster, name in [(0, "methane"), (1, "ethene")]:
            labels = ci.index[ci["cluster_number"] == cluster]
            assert df.iloc[-1][labels].sum() > df.iloc[0][labels].sum(), (
                f"{name} should form during pyrolysis"
            )

    def test_propane_delta_defined_at_final_time(self, sim_result):
        """δ¹³C of propane at the final time must be a finite number."""
        _, df, ci = sim_result
        delta = get_delta(df.iloc[-1], ci, cluster_num=5)
        assert np.isfinite(delta)
        assert -200 < delta < 200  # physically reasonable range

    def test_ethene_delta_defined_at_final_time(self, sim_result):
        """δ¹³C of ethene at the final time must be a finite number."""
        _, df, ci = sim_result
        delta = get_delta(df.iloc[-1], ci, cluster_num=1)
        assert np.isfinite(delta)
        assert -200 < delta < 200

    def test_methane_delta_defined_at_final_time(self, sim_result):
        """δ¹³C of methane at the final time must be a finite number."""
        _, df, ci = sim_result
        delta = get_delta(df.iloc[-1], ci, cluster_num=0)
        assert np.isfinite(delta)
        assert -200 < delta < 200

    def test_initial_propane_delta_near_minus28(self, sim_result):
        """δ¹³C of propane at first time step must still be close to −28‰."""
        _, df, ci = sim_result
        delta = get_delta(df.iloc[0], ci, cluster_num=5)
        # At t ≈ 1e-4 s very little has converted, so δ ≈ −28‰
        assert abs(delta - (-28.0)) < 2.0
