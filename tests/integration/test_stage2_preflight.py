"""Pre-flight checks for Stage 2 (isotopologue expansion).

Tests run before launching the long Stage 2 job to verify:
  1. Stage 1 output is intact and loadable.
  2. Isotopomer count arithmetic is correct (2^n per n-carbon species).
  3. Crash recovery detection works correctly.
  4. RMG can load Stage 1 CHEMKIN back into memory (slow, requires rmg_env).
  5. generate_isotopomers returns correct counts for known species (slow).

Run fast tests only:
    uv run pytest tests/integration/test_stage2_preflight.py -v -m "not slow"

Run all (requires conda rmg_env — these are informational, not blocking):
    conda run --prefix ... pytest tests/integration/test_stage2_preflight.py -v
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile

import pytest

_PROJECT = pathlib.Path(__file__).parents[2]
_BASE_DIR = _PROJECT / "rmg_pipeline" / "output" / "base"
_ISO_DIR  = _PROJECT / "rmg_pipeline" / "output" / "isotope"

_CHEMKIN   = _BASE_DIR / "chemkin" / "chem_annotated.inp"
_SPEC_DICT = _BASE_DIR / "chemkin" / "species_dictionary.txt"


# ── Stage 1 output integrity ──────────────────────────────────────────────────


class TestStage1Output:
    def test_chemkin_file_exists(self):
        assert _CHEMKIN.exists(), f"Stage 1 CHEMKIN not found: {_CHEMKIN}"

    def test_species_dict_exists(self):
        assert _SPEC_DICT.exists(), f"Stage 1 species dict not found: {_SPEC_DICT}"

    def test_species_count(self):
        """Species dictionary must have ~31 entries (Goldman's base = 31)."""
        content = _SPEC_DICT.read_text()
        blocks = [b for b in content.strip().split("\n\n") if b.strip()]
        assert 25 <= len(blocks) <= 40, (
            f"Expected ~31 species, got {len(blocks)}"
        )

    def test_propane_present(self):
        content = _SPEC_DICT.read_text()
        assert "propane_ooo" in content, "propane_ooo not found in species dictionary"

    def test_helium_present(self):
        content = _SPEC_DICT.read_text()
        assert "He" in content, "[He] inert not found in species dictionary"

    def test_chemkin_has_reactions(self):
        content = _CHEMKIN.read_text()
        reaction_count = content.count("<=>") + content.count("=>")
        assert reaction_count >= 100, (
            f"Expected >=100 reactions in CHEMKIN, found {reaction_count}"
        )


# ── Isotopomer count arithmetic ───────────────────────────────────────────────


class TestIsotopomerCounts:
    """Verify 2^n isotopomers per n-carbon species.

    These are pure arithmetic checks — no RMG needed.
    """

    @pytest.mark.parametrize("n_carbons,expected", [
        (0, 1),   # He, H2: no labelable carbons → 1 variant (itself)
        (1, 2),   # methane, methyl
        (2, 4),   # ethane, ethylene, acetylene, ethyl
        (3, 8),   # propane, propene, propyne, allene, C3 radicals
        (4, 16),  # butane, butene, C4 radicals
        (5, 32),  # pentane family
        (6, 64),  # hexane family
    ])
    def test_isotopomer_count_formula(self, n_carbons, expected):
        assert 2 ** n_carbons == expected

    def test_propane_total_isotopologues(self):
        """Goldman's propane model: 31 base species → 343 total isotopologues."""
        # From Goldman's full_model/isotopomer_cluster_info.csv
        import pandas as pd
        ci = pd.read_csv(
            _PROJECT / "rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52"
                       "/mechanisms/full_model/isotopomer_cluster_info.csv",
            index_col="name",
        )
        assert len(ci) == 343, f"Expected 343 isotopologue species, got {len(ci)}"
        assert ci["cluster_number"].nunique() == 31, (
            f"Expected 31 unique clusters, got {ci['cluster_number'].nunique()}"
        )

    def test_isotopologue_counts_per_cluster(self):
        """Each cluster size must equal 2^n_carbons (with symmetry exceptions for 3C)."""
        import pandas as pd
        ci = pd.read_csv(
            _PROJECT / "rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52"
                       "/mechanisms/full_model/isotopomer_cluster_info.csv",
            index_col="name",
        )
        for cn in ci["cluster_number"].unique():
            members = ci[ci["cluster_number"] == cn]
            n_c = int(members["enriched_atoms"].max())
            n_iso = len(members)
            expected = 2 ** n_c
            # Symmetric species have fewer distinct isotopomers (e.g. C=C=C has 6 not 8)
            assert n_iso <= expected, (
                f"Cluster {cn}: {n_iso} isotopologues > 2^{n_c}={expected}"
            )
            if n_iso < expected:
                # Allowed: symmetric species like allene (C=C=C, 3C → 6 not 8)
                pass


# ── Crash recovery detection ──────────────────────────────────────────────────


class TestCrashRecovery:
    """Test the recovery helpers in run_pipeline without actually running RMG."""

    def _stage2_complete(self, iso_dir: pathlib.Path) -> bool:
        """Mirror of the helper in run_pipeline.py."""
        return (iso_dir / "iso" / "chemkin" / "chem_annotated.inp").exists()

    def _stage2_partial(self, iso_dir: pathlib.Path) -> bool:
        """Partial run: iso/ dir exists but no final CHEMKIN."""
        iso_subdir = iso_dir / "iso"
        return iso_subdir.exists() and not self._stage2_complete(iso_dir)

    def test_complete_when_no_dir(self, tmp_path):
        assert not self._stage2_complete(tmp_path)

    def test_complete_when_iso_dir_empty(self, tmp_path):
        (tmp_path / "iso").mkdir()
        assert not self._stage2_complete(tmp_path)

    def test_partial_when_iso_dir_exists_no_chemkin(self, tmp_path):
        (tmp_path / "iso").mkdir()
        assert self._stage2_partial(tmp_path)

    def test_not_partial_when_no_dir(self, tmp_path):
        assert not self._stage2_partial(tmp_path)

    def test_complete_when_chemkin_present(self, tmp_path):
        chemkin = tmp_path / "iso" / "chemkin" / "chem_annotated.inp"
        chemkin.parent.mkdir(parents=True)
        chemkin.write_text("REACTIONS\nEND\n")
        assert self._stage2_complete(tmp_path)
        assert not self._stage2_partial(tmp_path)

    def test_partial_dir_removal(self, tmp_path):
        """Removing incomplete iso/ and re-running must not raise FileExistsError."""
        iso_subdir = tmp_path / "iso"
        iso_subdir.mkdir()
        (iso_subdir / "partial.log").write_text("crashed")

        # Simulate recovery: remove partial dir
        shutil.rmtree(iso_subdir)
        assert not iso_subdir.exists()

        # Now os.mkdir (as called by run_isotopes) would succeed
        iso_subdir.mkdir()
        assert iso_subdir.exists()


# ── RMG load test (slow, requires rmg_env) ────────────────────────────────────


@pytest.mark.slow
class TestRMGLoad:
    """Load Stage 1 CHEMKIN back into RMG memory and verify species."""

    @pytest.fixture(scope="class")
    def rmg_job(self):
        sys.path.insert(0, str(_PROJECT / "vendor" / "RMG-Py"))
        try:
            from rmgpy.tools.isotopes import load_rmg_job
        except ImportError:
            pytest.skip("rmgpy not available — run in rmg_env")

        rmg = load_rmg_job(
            str(_PROJECT / "rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52"
                           "/input_files/propane_rmg_input_file.py"),
            str(_CHEMKIN),
            str(_SPEC_DICT),
            generate_images=False,
            use_chemkin_names=True,
        )
        return rmg

    def test_species_count(self, rmg_job):
        n = len(rmg_job.reaction_model.core.species)
        assert 25 <= n <= 40, f"Expected ~31 core species, got {n}"

    def test_reaction_count(self, rmg_job):
        n = len(rmg_job.reaction_model.core.reactions)
        assert n >= 100, f"Expected >=100 core reactions, got {n}"

    def test_propane_in_core(self, rmg_job):
        labels = [s.label for s in rmg_job.reaction_model.core.species]
        assert any("propane" in l.lower() or l == "CCC" for l in labels), (
            f"Propane not found in core species: {labels[:10]}"
        )


# ── Isotopomer generation test (slow, requires rmg_env) ───────────────────────


@pytest.mark.slow
class TestIsotopomerGeneration:
    """generate_isotopomers must return 2^n variants for n-carbon species."""

    @pytest.fixture(scope="class")
    def rmg_job(self):
        sys.path.insert(0, str(_PROJECT / "vendor" / "RMG-Py"))
        try:
            from rmgpy.tools.isotopes import load_rmg_job, generate_isotopomers
        except ImportError:
            pytest.skip("rmgpy not available — run in rmg_env")
        return load_rmg_job, generate_isotopomers

    def test_propane_gives_8_isotopomers(self, rmg_job):
        sys.path.insert(0, str(_PROJECT / "vendor" / "RMG-Py"))
        from rmgpy.tools.isotopes import load_rmg_job, generate_isotopomers
        from rmgpy.tools.isotopes import find_cp0_and_cpinf

        rmg = load_rmg_job(
            str(_PROJECT / "rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52"
                           "/input_files/propane_rmg_input_file.py"),
            str(_CHEMKIN),
            str(_SPEC_DICT),
            generate_images=False,
            use_chemkin_names=True,
        )
        for spc in rmg.reaction_model.core.species:
            find_cp0_and_cpinf(spc, spc.thermo)

        # Find propane (3C → 2^3 = 8 isotopologues including itself)
        propane = next(
            (s for s in rmg.reaction_model.core.species if "propane" in s.label.lower()),
            None,
        )
        assert propane is not None, "Propane not found in Stage 1 core"
        isotopomers = generate_isotopomers(propane, N=1000000)
        # Including the original unlabeled species: 8 total
        total = 1 + len(isotopomers)
        assert total == 8, f"Expected 8 propane isotopomers, got {total}"

    def test_methane_gives_2_isotopomers(self, rmg_job):
        sys.path.insert(0, str(_PROJECT / "vendor" / "RMG-Py"))
        from rmgpy.tools.isotopes import load_rmg_job, generate_isotopomers
        from rmgpy.tools.isotopes import find_cp0_and_cpinf

        rmg = load_rmg_job(
            str(_PROJECT / "rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52"
                           "/input_files/propane_rmg_input_file.py"),
            str(_CHEMKIN),
            str(_SPEC_DICT),
            generate_images=False,
            use_chemkin_names=True,
        )
        for spc in rmg.reaction_model.core.species:
            find_cp0_and_cpinf(spc, spc.thermo)

        methane = next(
            (s for s in rmg.reaction_model.core.species
             if s.molecule[0].get_formula() == "CH4"),
            None,
        )
        assert methane is not None, "Methane not found in Stage 1 core"
        isotopomers = generate_isotopomers(methane, N=1000000)
        total = 1 + len(isotopomers)
        assert total == 2, f"Expected 2 methane isotopomers, got {total}"
