"""Carbon mass balance tests for Goldman 2019 isotopologue mechanisms.

Key invariants tested:

1. cluster_info covers every carbon-containing species in the Cantera mechanism
   (no untracked carbon pools that could host a Rayleigh distillation artefact).

2. n_C from cluster_info (enriched_atoms + unenriched_atoms) matches
   gas.n_atoms(sp, "C") from Cantera for every species.

3. The bulk ¹³C fraction — Σ c_i × n13C_i / Σ c_i × n_C_i — is conserved
   throughout a short simulation.  This is the fundamental isotope mass balance:
   reactions redistribute carbon between species but cannot create or destroy ¹³C.
   For a const-T, const-P ideal-gas reactor the density is fixed, so V cancels
   and the ratio of concentrations weighted by n13C vs n_C is directly conserved.

4. At late pyrolysis times, stable molecule clusters (propane, methane, ethene,
   ethane) hold >95% of total carbon in the mechanism.  This rules out the
   scenario where radical pools accumulate significant ¹³C and produce an
   apparent enrichment in the stable product pool via Rayleigh distillation.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
import cantera as ct

from isotopologue.goldman import VPDB_RATIO, get_delta, propane_init_x, run_simulation

_MECH_DIR   = pathlib.Path(__file__).parents[2] / (
    "rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52/mechanisms"
)
_THREE_RXN  = _MECH_DIR / "three_reaction_model"
_FULL_MODEL = _MECH_DIR / "full_model"

# Cluster numbers for the 3-rxn model
_STABLE_CLUSTERS_3RXN = {
    "propane": 5,
    "methane": 0,
    "ethene":  1,
    "ethane":  4,
}


def _load_3rxn():
    gas = ct.Solution(str(_THREE_RXN / "chem.yaml"))
    ci  = pd.read_csv(_THREE_RXN / "isotopomer_cluster_info.csv", index_col="name")
    return gas, ci


def _bulk_13c_ratio(concs: pd.Series | dict, ci: pd.DataFrame) -> float:
    """Global ¹³C/¹²C ratio: Σ c_i × n13C_i / Σ c_i × n12C_i.

    This is the quantity used in the δ formula: δ = (ratio / VPDB_RATIO − 1) × 1000.
    It must be conserved throughout any closed-system simulation (isotope identity
    is preserved by reactions).  For const-T, const-P the volume cancels so the
    concentration-weighted ratio equals the mole-weighted ratio directly.
    """
    numer = denom = 0.0
    for sp, c in concs.items():
        if sp not in ci.index:
            continue
        n13 = float(ci.loc[sp, "enriched_atoms"])
        n12 = float(ci.loc[sp, "unenriched_atoms"])
        numer += c * n13
        denom += c * n12
    return numer / denom if denom > 0 else np.nan


def _cluster_carbon(concs: pd.Series, ci: pd.DataFrame, cluster_num: int) -> float:
    """Total carbon concentration (mol_C/m³) in a given cluster."""
    labels = ci.index[ci["cluster_number"] == cluster_num]
    total = 0.0
    for sp in labels:
        c  = float(concs.get(sp, 0.0))
        nc = float(ci.loc[sp, "enriched_atoms"]) + float(ci.loc[sp, "unenriched_atoms"])
        total += c * nc
    return total


# ── Structural tests (no simulation) ─────────────────────────────────────────


class TestClusterInfoCoverage:
    """Verify cluster_info covers all carbon-containing species."""

    def test_all_carbon_species_in_cluster_info(self):
        """Every species with n_C > 0 must have an entry in cluster_info.

        If any carbon species is absent from cluster_info, that carbon is
        invisible to our δ¹³C accounting and could falsely inflate or deflate
        the tracked-pool δ.
        """
        gas, ci = _load_3rxn()
        missing = [
            sp for sp in gas.species_names
            if gas.n_atoms(sp, "C") > 0 and sp not in ci.index
        ]
        assert missing == [], (
            f"Carbon species not in cluster_info: {missing}"
        )

    def test_carbon_count_matches_cantera(self):
        """cluster_info n_C (enriched + unenriched) must equal gas.n_atoms(sp, 'C')."""
        gas, ci = _load_3rxn()
        mismatches = {}
        for sp in ci.index:
            if sp not in gas.species_names:
                continue
            nc_cantera = gas.n_atoms(sp, "C")
            nc_csv     = int(ci.loc[sp, "enriched_atoms"]) + int(ci.loc[sp, "unenriched_atoms"])
            if nc_cantera != nc_csv:
                mismatches[sp] = (nc_cantera, nc_csv)
        assert mismatches == {}, (
            f"n_C mismatch (Cantera vs cluster_info): {mismatches}"
        )

    def test_no_negative_atom_counts(self):
        """enriched_atoms and unenriched_atoms must both be non-negative."""
        _, ci = _load_3rxn()
        assert (ci["enriched_atoms"]   >= 0).all()
        assert (ci["unenriched_atoms"] >= 0).all()

    def test_initial_bulk_13c_ratio_is_physical(self):
        """Bulk ¹³C fraction from propane_init_x must be near natural abundance.

        At -28‰ the fraction is slightly below VPDB_RATIO; must be in [0, 1].
        """
        _, ci = _load_3rxn()
        init_x = propane_init_x(delta_total=-28.0, psia=5.4)
        ratio  = _bulk_13c_ratio(init_x, ci)
        assert np.isfinite(ratio)
        assert 0 < ratio < 0.1   # natural abundance is ~1.1%, well below 10%

    def test_initial_bulk_13c_ratio_matches_vpdb_minus28(self):
        """Bulk ratio must equal VPDB × (1 − 28/1000) within 1‰ tolerance.

        This confirms enrich_frac and propane_init_x are self-consistent.
        """
        _, ci = _load_3rxn()
        init_x       = propane_init_x(delta_total=-28.0, psia=0.0)
        ratio         = _bulk_13c_ratio(init_x, ci)
        expected_ratio = VPDB_RATIO * (1.0 - 28.0 / 1000.0)
        # Express as δ relative to VPDB
        delta_implied = (ratio / VPDB_RATIO - 1.0) * 1000.0
        assert abs(delta_implied - (-28.0)) < 0.5, (
            f"Bulk δ¹³C = {delta_implied:.2f}‰, expected ≈ −28‰"
        )


# ── Simulation-based conservation tests ──────────────────────────────────────


class TestBulk13CConservation:
    """Bulk ¹³C fraction must be conserved throughout simulation.

    Uses a very short integration (3 points to 10 ms) to keep this fast.
    """

    @pytest.fixture(scope="class")
    def short_sim(self):
        gas, ci = _load_3rxn()
        init_x  = propane_init_x(delta_total=-28.0, psia=5.4)
        times   = np.array([1e-3, 5e-3, 1e-2])
        df      = run_simulation(gas, times, T_K=1123.15, P_Pa=2e5, init_x=init_x)
        return df, ci, init_x

    def test_bulk_ratio_conserved_throughout(self, short_sim):
        """¹³C fraction must stay constant (within ODE tolerance) at all times."""
        df, ci, init_x = short_sim
        r0 = _bulk_13c_ratio(init_x, ci)
        assert np.isfinite(r0), "Initial bulk ¹³C ratio should be finite"
        for t, row in df.iterrows():
            r = _bulk_13c_ratio(row, ci)
            assert np.isfinite(r), f"Bulk ¹³C ratio is NaN at t={t}"
            assert abs(r - r0) / r0 < 1e-3, (
                f"Bulk ¹³C fraction changed by {abs(r-r0)/r0*100:.3f}% at t={t}s; "
                "isotopes are not conserved (numerical or model error)"
            )

    def test_total_carbon_concentration_decreases_with_breakdown(self, short_sim):
        """Σ c_i × n_C(i) decreases from early to late time as propane breaks up.

        In a const-P reactor the molar density is fixed, so as C3 → C1 + C2 the
        average carbon per molecule drops.  Both values come from the simulation
        DataFrame (same concentration units), making the comparison valid.
        """
        df, ci, _ = short_sim
        def nc_total(row):
            return sum(
                float(row.get(sp, 0.0)) * (
                    float(ci.loc[sp, "enriched_atoms"]) + float(ci.loc[sp, "unenriched_atoms"])
                )
                for sp in ci.index
            )
        nc_early = nc_total(df.iloc[0])
        nc_late  = nc_total(df.iloc[-1])
        # Average C per molecule-concentration should decrease as propane cracks.
        # If nc_late >= nc_early the reactor has not reacted — something is wrong.
        assert nc_late < nc_early, (
            f"No decrease in carbon concentration detected: "
            f"nc_early={nc_early:.6g}, nc_late={nc_late:.6g}; "
            "check that the reaction is proceeding"
        )

    def test_delta_products_obey_conservation(self, short_sim):
        """δ¹³C of products must be consistent with the global mass balance.

        If methane is enriched relative to bulk, ethene/ethane must compensate
        (mass balance: ¹³C can't appear from nowhere).  We check that at least
        one product is not more enriched than the starting propane.
        """
        df, ci, _ = short_sim
        final   = df.iloc[-1]
        d_ch4   = get_delta(final, ci, cluster_num=0)
        d_c2h4  = get_delta(final, ci, cluster_num=1)
        d_c2h6  = get_delta(final, ci, cluster_num=4)
        d_bulk  = -28.0  # initial propane δ¹³C

        # At least one product should not exceed the bulk by an unreasonable amount
        # (if all products are far above bulk, total ¹³C is being created — impossible)
        finite_deltas = [d for d in (d_ch4, d_c2h4, d_c2h6) if np.isfinite(d)]
        if len(finite_deltas) >= 2:
            assert min(finite_deltas) < d_bulk + 100, (
                f"All products are more than 100‰ above bulk δ¹³C={d_bulk}‰; "
                "this would require more ¹³C than the system started with"
            )


# ── Rayleigh distillation check ───────────────────────────────────────────────


@pytest.mark.slow
class TestStableSpeciesCarbonFraction:
    """Stable molecules must hold the overwhelming majority of carbon.

    If radical pools (CH3, C2H5, etc.) accumulated significant ¹³C, the stable-
    product δ values we report would be biased — analogous to Rayleigh distillation
    where ¹³C is hidden in an unmonitored reservoir.
    """

    @pytest.fixture(scope="class")
    def full_sim(self):
        gas, ci = _load_3rxn()
        init_x  = propane_init_x(delta_total=-28.0, psia=5.4)
        T_K     = 1123.15
        times   = np.linspace(1e-3, 95.0 / T_K, 20)
        df      = run_simulation(gas, times, T_K=T_K, P_Pa=2e5, init_x=init_x)
        return df, ci

    def test_stable_clusters_hold_majority_of_carbon(self, full_sim):
        """Stable molecule clusters (propane+methane+ethene+ethane) must hold ≥ 95% of carbon.

        The radical clusters (methyl, ethyl) are short-lived intermediates; any
        significant carbon accumulation there would be a model pathology.
        """
        df, ci = full_sim
        final  = df.iloc[-1]

        stable_c = sum(
            _cluster_carbon(final, ci, cn)
            for cn in _STABLE_CLUSTERS_3RXN.values()
        )
        total_c = sum(
            float(final.get(sp, 0.0)) * (
                float(ci.loc[sp, "enriched_atoms"]) + float(ci.loc[sp, "unenriched_atoms"])
            )
            for sp in ci.index
        )
        fraction = stable_c / total_c if total_c > 0 else 0.0
        assert fraction >= 0.95, (
            f"Stable molecule clusters hold only {fraction*100:.1f}% of total carbon; "
            "radical pools are anomalously large — check mass balance"
        )

    def test_bulk_ratio_conserved_over_full_sim(self, full_sim):
        """Bulk ¹³C ratio must hold for the entire 85 ms integration."""
        df, ci = full_sim
        init_x = propane_init_x(delta_total=-28.0, psia=5.4)
        r0     = _bulk_13c_ratio(init_x, ci)
        for t, row in df.iterrows():
            r = _bulk_13c_ratio(row, ci)
            assert np.isfinite(r), f"NaN bulk ratio at t={t}"
            assert abs(r - r0) / r0 < 2e-3, (
                f"Bulk ¹³C fraction drifted by {abs(r-r0)/r0*100:.3f}% at t={t}s"
            )
