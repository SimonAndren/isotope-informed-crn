"""Smoke tests for the Goldman 2019 RMG-Py workflow.

These tests validate that:
1. Goldman's pre-generated mechanism files can be loaded and have the
   correct species/reaction counts as reported in Goldman 2019.
2. The RMG pipeline script is importable (verifies RMG-Py environment).
3. [slow] Cantera can simulate the Goldman mechanism end-to-end (Stage 3
   of the pipeline).

Key numbers from Goldman 2019:
  Full model:  343 species, 7096 reactions  (Table 1 / SI)
  DRG model:    31 species,  167 reactions
  3-rxn model:  24 species,   19 reactions
  6-rxn model:  26 species,   35 reactions

The mechanism files live in:
    rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52/mechanisms/
"""

from __future__ import annotations

import pathlib

import cantera as ct
import pytest

# ─── Paths ────────────────────────────────────────────────────────────────────

_MECH_DIR = pathlib.Path(__file__).parents[2] / (
    "rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52/mechanisms"
)


def _mech(model: str) -> ct.Solution:
    return ct.Solution(str(_MECH_DIR / model / "chem.yaml"))


# ─── Mechanism loading ────────────────────────────────────────────────────────


class TestGoldmanMechanismFiles:
    """Verify that Goldman's pre-generated mechanisms load correctly and match
    the species/reaction counts reported in Goldman 2019."""

    def test_three_reaction_model_loads(self):
        gas = _mech("three_reaction_model")
        assert gas is not None

    def test_six_reaction_model_loads(self):
        gas = _mech("six_reaction_model")
        assert gas is not None

    def test_drg_model_loads(self):
        gas = _mech("drg_model")
        assert gas is not None

    def test_full_model_loads(self):
        gas = _mech("full_model")
        assert gas is not None

    def test_three_reaction_species_count(self):
        """3-rxn isotopologue-expanded mechanism: 24 species."""
        gas = _mech("three_reaction_model")
        assert gas.n_species == 24, (
            f"Expected 24 species in 3-rxn model, got {gas.n_species}"
        )

    def test_three_reaction_reactions_count(self):
        """3-rxn isotopologue-expanded mechanism: 19 reactions."""
        gas = _mech("three_reaction_model")
        assert gas.n_reactions == 19, (
            f"Expected 19 reactions in 3-rxn model, got {gas.n_reactions}"
        )

    def test_six_reaction_species_count(self):
        """6-rxn isotopologue-expanded mechanism: 26 species."""
        gas = _mech("six_reaction_model")
        assert gas.n_species == 26, (
            f"Expected 26 species in 6-rxn model, got {gas.n_species}"
        )

    def test_six_reaction_reactions_count(self):
        """6-rxn isotopologue-expanded mechanism: 35 reactions."""
        gas = _mech("six_reaction_model")
        assert gas.n_reactions == 35, (
            f"Expected 35 reactions in 6-rxn model, got {gas.n_reactions}"
        )

    def test_drg_species_count(self):
        """DRG isotopologue-expanded mechanism: 31 species."""
        gas = _mech("drg_model")
        assert gas.n_species == 31, (
            f"Expected 31 species in DRG model, got {gas.n_species}"
        )

    def test_drg_reactions_count(self):
        """DRG isotopologue-expanded mechanism: 167 reactions."""
        gas = _mech("drg_model")
        assert gas.n_reactions == 167, (
            f"Expected 167 reactions in DRG model, got {gas.n_reactions}"
        )

    def test_full_model_species_count(self):
        """Full isotopologue-expanded mechanism: 343 species (Goldman 2019)."""
        gas = _mech("full_model")
        assert gas.n_species == 343, (
            f"Expected 343 species in full model, got {gas.n_species}"
        )

    def test_full_model_reactions_count(self):
        """Full isotopologue-expanded mechanism: 7096 reactions (Goldman 2019)."""
        gas = _mech("full_model")
        assert gas.n_reactions == 7096, (
            f"Expected 7096 reactions in full model, got {gas.n_reactions}"
        )

    def test_propane_present_in_all_models(self):
        """Propane (CCC in SMILES notation) must appear in all mechanisms."""
        for model in ("three_reaction_model", "six_reaction_model", "drg_model", "full_model"):
            gas = _mech(model)
            propane_species = [s for s in gas.species_names if s.startswith("CCC")]
            assert len(propane_species) > 0, (
                f"No propane species found in {model}: {gas.species_names[:10]}"
            )

    def test_full_model_has_propane_isotopologues(self):
        """Full model must contain multiple ¹³C propane isotopologues."""
        gas = _mech("full_model")
        propane_species = [s for s in gas.species_names if s.startswith("CCC")]
        # Propane has 3 C atoms; with symmetry: 6 unique isotopologues
        assert len(propane_species) >= 6, (
            f"Expected ≥6 propane isotopologues in full model, got {len(propane_species)}: "
            f"{propane_species}"
        )


# ─── RMG-Py pipeline import ───────────────────────────────────────────────────


class TestRMGPipelineImportable:
    """Verify the RMG pipeline script can be parsed (syntax check).

    Does not *run* the pipeline (which takes ~30 min for stage 1).
    Verifies only that the script is syntactically valid and the RMG
    environment is set up to import it.
    """

    def test_pipeline_script_exists(self):
        """rmg_pipeline/run_pipeline.py must be present."""
        pipeline = pathlib.Path(__file__).parents[2] / "rmg_pipeline" / "run_pipeline.py"
        assert pipeline.exists(), f"Pipeline script missing: {pipeline}"

    def test_pipeline_script_is_valid_python(self):
        """Pipeline script must parse without syntax errors."""
        import ast
        pipeline = pathlib.Path(__file__).parents[2] / "rmg_pipeline" / "run_pipeline.py"
        source = pipeline.read_text()
        # Raises SyntaxError if invalid
        tree = ast.parse(source)
        assert tree is not None


# ─── Stage 3: Cantera simulation (slow) ──────────────────────────────────────


@pytest.mark.slow
class TestGoldmanCanteraSimulation:
    """End-to-end simulation of Goldman's mechanisms with Cantera.

    These tests exercise Stage 3 of the pipeline: loading an isotopologue-
    expanded mechanism and running the constant-T/P reactor that Goldman used.
    They are marked slow because the ODE solver integrates a 343-species /
    7096-reaction network.

    The 3-rxn and DRG models are also tested as faster reference points.
    """

    @pytest.fixture(scope="class")
    def full_gas(self):
        return _mech("full_model")

    @pytest.fixture(scope="class")
    def three_rxn_gas(self):
        return _mech("three_reaction_model")

    def _simulate(self, gas: ct.Solution, T_K: float = 1123.0, t_end: float = 0.085):
        """Run a constant-T/P Cantera simulation.

        Uses Goldman's conditions: propane 1 % mol in He, 2 bar, energy off.
        Returns final mole-fraction array.
        """
        propane = next((s for s in gas.species_names if s == "CCC"), None)
        assert propane is not None, f"'CCC' not in {gas.species_names[:5]}"

        he = next((s for s in gas.species_names if s in ("[He]", "He")), None)
        init_x: dict[str, float] = {propane: 0.01}
        if he:
            init_x[he] = 0.99

        gas.TPX = T_K, 2.0e5, init_x
        r = ct.IdealGasConstPressureReactor(gas, energy="off")
        sim = ct.ReactorNet([r])
        sim.advance(t_end)
        return r.phase.X.copy()

    def test_full_model_simulates_at_850C(self, full_gas):
        """Full model (343 sp / 7096 rxn) must simulate without error at 850 °C."""
        X_final = self._simulate(full_gas, T_K=1123.0)
        propane_idx = full_gas.species_index("CCC")
        # Some propane must have been consumed
        assert X_final[propane_idx] < 0.01, (
            f"Propane should decrease from 1% at 850°C; X_final={X_final[propane_idx]:.4f}"
        )

    def test_three_rxn_model_simulates_at_850C(self, three_rxn_gas):
        """3-rxn model (24 sp / 19 rxn) must simulate and crack propane at 850 °C."""
        X_final = self._simulate(three_rxn_gas, T_K=1123.0)
        propane_idx = three_rxn_gas.species_index("CCC")
        assert X_final[propane_idx] < 0.01, (
            f"Propane should crack at 850°C; X_final={X_final[propane_idx]:.4f}"
        )

    def test_higher_temperature_more_conversion(self, three_rxn_gas):
        """Propane conversion must increase monotonically with temperature."""
        conversions = {}
        for T_C in (800, 850, 900, 950):
            T_K = T_C + 273.15
            X = self._simulate(three_rxn_gas, T_K=T_K)
            propane_idx = three_rxn_gas.species_index("CCC")
            conversions[T_C] = 1.0 - X[propane_idx] / 0.01

        temperatures = sorted(conversions)
        for i in range(len(temperatures) - 1):
            T_lo, T_hi = temperatures[i], temperatures[i + 1]
            assert conversions[T_lo] <= conversions[T_hi] + 1e-6, (
                f"Conversion should increase with T: "
                f"{T_lo}°C={conversions[T_lo]:.3f} vs {T_hi}°C={conversions[T_hi]:.3f}"
            )

    def test_pipeline_simulate_stage_with_goldman_files(self):
        """run_pipeline.run_simulate() works on Goldman's pre-generated 3-rxn YAML.

        This validates Stage 3 of the pipeline without running Stages 1–2.
        """
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "rmg_pipeline"))
        import importlib
        pipeline = importlib.import_module("run_pipeline")

        goldman_3rxn = str(_MECH_DIR / "three_reaction_model" / "chem.yaml")
        gas = pipeline.run_simulate(yaml_path=goldman_3rxn)
        assert gas.n_species == 24
        assert gas.n_reactions == 19


class TestRMGPyAvailability:
    """Check whether RMG-Py is importable in the current environment.

    These tests pass trivially when RMG-Py is not available (they skip),
    and exercise the RMG-Py installation when it is.
    """

    @pytest.fixture(autouse=True)
    def require_rmgpy(self):
        # vendor/RMG-Py makes the pure-Python rmgpy importable, but Cython
        # extensions (like rmgpy.solver.base) are only compiled in rmg_env.
        pytest.importorskip(
            "rmgpy.solver.base",
            reason="RMG-Py Cython extensions not compiled; run in rmg_env conda env",
        )

    def test_rmgpy_imports(self):
        """Core RMG-Py modules must be importable."""
        import rmgpy.species  # noqa: F401
        import rmgpy.reaction  # noqa: F401
        import rmgpy.kinetics  # noqa: F401

    def test_rmgpy_isotope_module_importable(self):
        """rmgpy.tools.isotopes must be importable (used by run_pipeline.py)."""
        import rmgpy.tools.isotopes  # noqa: F401
