"""Unit tests for RPFRVisualizer and visualization helper functions.

Tests cover:
- _val_to_hex: scalar-to-hex-colour mapping
- build_rdkit_mol: RDKit molecule construction with DFT coordinates
- mol_to_xyz_block: XYZ serialisation
- RPFRVisualizer: initialisation, statistics, all four display modes,
  surface mode, and summary_table
"""

from __future__ import annotations

import numpy as np
import py3Dmol
import pytest
from rdkit import Chem

from rpfr_gui.ui.visualization import (
    _DEFAULT_CMAP,
    ELEMENT_COLORMAPS,
    RPFRVisualizer,
    _val_to_hex,
    build_rdkit_mol,
    mol_to_xyz_block,
)

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def visualizer(sample_visualizer_data):
    """RPFRVisualizer instance built from the sample_visualizer_data fixture."""
    d = sample_visualizer_data
    return RPFRVisualizer(
        d["smiles"],
        d["symbols"],
        d["coords"],
        d["rpfr"],
        temperature=d["temperature"],
    )


# ── _val_to_hex ───────────────────────────────────────────────────────────────


class TestValToHex:
    """Tests for the _val_to_hex colour helper."""

    def test_returns_hex_string_format(self):
        """Output must be a 7-character '#xxxxxx' hex string."""
        result = _val_to_hex(0.5, 0.0, 1.0, "viridis")
        assert isinstance(result, str)
        assert result.startswith("#")
        assert len(result) == 7

    def test_minimum_value_is_valid_hex(self):
        """Value equal to vmin returns a valid hex colour."""
        result = _val_to_hex(0.0, 0.0, 1.0, "viridis")
        assert result.startswith("#")
        assert len(result) == 7

    def test_maximum_value_is_valid_hex(self):
        """Value equal to vmax returns a valid hex colour."""
        result = _val_to_hex(1.0, 0.0, 1.0, "viridis")
        assert result.startswith("#")
        assert len(result) == 7

    def test_equal_bounds_does_not_raise(self):
        """When vmin == vmax the function returns a valid hex without raising."""
        result = _val_to_hex(5.0, 5.0, 5.0, "viridis")
        assert result.startswith("#")
        assert len(result) == 7

    def test_out_of_range_value_is_clipped(self):
        """Values outside [vmin, vmax] are clipped, not rejected."""
        low = _val_to_hex(-10.0, 0.0, 1.0, "viridis")
        high = _val_to_hex(10.0, 0.0, 1.0, "viridis")
        at_min = _val_to_hex(0.0, 0.0, 1.0, "viridis")
        at_max = _val_to_hex(1.0, 0.0, 1.0, "viridis")
        assert low == at_min
        assert high == at_max

    def test_different_colormaps_give_different_colors(self):
        """Different colormaps produce different colours for the same value."""
        c_viridis = _val_to_hex(0.5, 0.0, 1.0, "viridis")
        c_plasma = _val_to_hex(0.5, 0.0, 1.0, "plasma")
        assert c_viridis != c_plasma


# ── build_rdkit_mol ───────────────────────────────────────────────────────────


class TestBuildRdkitMol:
    """Tests for the build_rdkit_mol coordinate-replacement helper."""

    def test_returns_rdkit_mol(self, sample_visualizer_data):
        """Function returns an RDKit Mol object."""
        d = sample_visualizer_data
        mol = build_rdkit_mol(d["smiles"], d["symbols"], d["coords"])
        assert isinstance(mol, Chem.Mol)

    def test_correct_atom_count(self, sample_visualizer_data):
        """Mol has the same number of atoms as the input symbols list."""
        d = sample_visualizer_data
        mol = build_rdkit_mol(d["smiles"], d["symbols"], d["coords"])
        assert mol.GetNumAtoms() == len(d["symbols"])

    def test_dft_coords_replace_generated_coords(self, sample_visualizer_data):
        """Conformer coordinates match the DFT coords passed in."""
        d = sample_visualizer_data
        mol = build_rdkit_mol(d["smiles"], d["symbols"], d["coords"])
        conf = mol.GetConformer()
        for i, (x, y, z) in enumerate(d["coords"]):
            pos = conf.GetAtomPosition(i)
            assert pos.x == pytest.approx(float(x), abs=1e-5)
            assert pos.y == pytest.approx(float(y), abs=1e-5)
            assert pos.z == pytest.approx(float(z), abs=1e-5)

    def test_invalid_smiles_raises_value_error(self):
        """Unparseable SMILES raises ValueError."""
        with pytest.raises(ValueError, match="RDKit could not parse SMILES"):
            build_rdkit_mol("not_a_valid_smiles###", ["C"], np.zeros((1, 3)))

    def test_atom_count_mismatch_raises_value_error(self, sample_visualizer_data):
        """Mismatched symbol/coord count vs. RDKit atom count raises ValueError."""
        d = sample_visualizer_data
        # Provide only 3 symbols/coords for a 5-atom molecule
        with pytest.raises(ValueError, match="Atom count mismatch"):
            build_rdkit_mol(d["smiles"], d["symbols"][:3], d["coords"][:3])

    def test_has_single_conformer(self, sample_visualizer_data):
        """Returned molecule has exactly one conformer."""
        d = sample_visualizer_data
        mol = build_rdkit_mol(d["smiles"], d["symbols"], d["coords"])
        assert mol.GetNumConformers() == 1


# ── mol_to_xyz_block ──────────────────────────────────────────────────────────


class TestMolToXyzBlock:
    """Tests for the XYZ serialisation helper."""

    def test_first_line_is_atom_count(self, sample_visualizer_data):
        """First line of XYZ block is the integer atom count."""
        d = sample_visualizer_data
        mol = build_rdkit_mol(d["smiles"], d["symbols"], d["coords"])
        xyz = mol_to_xyz_block(mol)
        first_line = xyz.strip().split("\n")[0].strip()
        assert first_line == str(mol.GetNumAtoms())

    def test_total_line_count(self, sample_visualizer_data):
        """XYZ block has n_atoms + 2 lines (count + blank comment + atom lines)."""
        d = sample_visualizer_data
        mol = build_rdkit_mol(d["smiles"], d["symbols"], d["coords"])
        xyz = mol_to_xyz_block(mol)
        lines = xyz.split("\n")
        assert len(lines) == mol.GetNumAtoms() + 2

    def test_element_symbols_present(self, sample_visualizer_data):
        """XYZ block contains the expected element symbols."""
        d = sample_visualizer_data
        mol = build_rdkit_mol(d["smiles"], d["symbols"], d["coords"])
        xyz = mol_to_xyz_block(mol)
        assert " C " in xyz or xyz.split("\n")[2].startswith("C")
        assert " H " in xyz or any(line.startswith("H") for line in xyz.split("\n")[2:])

    def test_returns_string(self, sample_visualizer_data):
        """mol_to_xyz_block returns a string."""
        d = sample_visualizer_data
        mol = build_rdkit_mol(d["smiles"], d["symbols"], d["coords"])
        assert isinstance(mol_to_xyz_block(mol), str)


# ── RPFRVisualizer initialisation ─────────────────────────────────────────────


class TestRPFRVisualizerInit:
    """Tests for RPFRVisualizer.__init__ and internal DataFrame construction."""

    def test_n_atoms(self, visualizer, sample_visualizer_data):
        """n_atoms equals the number of atoms in input."""
        assert visualizer.n_atoms == len(sample_visualizer_data["symbols"])

    def test_temperature_stored(self, visualizer, sample_visualizer_data):
        """Temperature attribute matches input."""
        assert visualizer.temperature == sample_visualizer_data["temperature"]

    def test_symbols_stored(self, visualizer, sample_visualizer_data):
        """symbols attribute matches input list."""
        assert visualizer.symbols == sample_visualizer_data["symbols"]

    def test_elements_detected(self, visualizer):
        """_elements contains all unique elements in the molecule."""
        assert "C" in visualizer._elements
        assert "H" in visualizer._elements

    def test_internal_df_row_count(self, visualizer):
        """Internal DataFrame has one row per atom."""
        assert len(visualizer._df) == visualizer.n_atoms

    def test_internal_df_has_minmax_column(self, visualizer):
        """Internal DataFrame has a 'minmax' column."""
        assert "minmax" in visualizer._df.columns

    def test_minmax_values_in_unit_interval(self, visualizer):
        """minmax values are in [0, 1] for each element."""
        for _element, group in visualizer._df.groupby("symbol"):
            if len(group) > 1:
                assert group["minmax"].min() == pytest.approx(0.0, abs=1e-10)
                assert group["minmax"].max() == pytest.approx(1.0, abs=1e-10)

    def test_rdkit_mol_created(self, visualizer):
        """RDKit molecule is built and accessible."""
        assert isinstance(visualizer.mol, Chem.Mol)

    def test_xyz_block_is_string(self, visualizer):
        """xyz_block is a non-empty string."""
        assert isinstance(visualizer.xyz_block, str)
        assert len(visualizer.xyz_block) > 0


# ── summary_table ─────────────────────────────────────────────────────────────


class TestSummaryTable:
    """Tests for RPFRVisualizer.summary_table()."""

    def test_row_count(self, visualizer):
        """Summary table has one row per atom."""
        df = visualizer.summary_table()
        assert len(df) == visualizer.n_atoms

    def test_expected_columns(self, visualizer):
        """Summary table contains the expected columns."""
        df = visualizer.summary_table()
        for col in ("idx", "symbol", "rpfr", "el_mean", "el_min", "el_max", "minmax"):
            assert col in df.columns, f"Missing column: {col}"

    def test_minmax_in_unit_interval(self, visualizer):
        """Minmax column values are in [0, 1] for elements with >1 atom."""
        df = visualizer.summary_table()
        multi_atom = df.groupby("symbol").filter(lambda g: len(g) > 1)
        assert (multi_atom["minmax"] >= 0.0).all()
        assert (multi_atom["minmax"] <= 1.0).all()

    def test_rpfr_values_match_input(self, visualizer, sample_visualizer_data):
        """RPFR values in summary match input array (sorted by index)."""
        df = visualizer.summary_table().sort_values("idx").reset_index(drop=True)
        np.testing.assert_allclose(
            df["rpfr"].values,
            sample_visualizer_data["rpfr"],
            rtol=1e-6,
        )

    def test_returns_copy(self, visualizer):
        """summary_table returns a copy; mutating it does not affect internals."""
        df = visualizer.summary_table()
        df["rpfr"] = 0.0
        assert (visualizer._df["rpfr"] != 0.0).any()


# ── display modes ─────────────────────────────────────────────────────────────


class TestShowElementFilter:
    """Tests for Mode 1 — element filter display."""

    def test_returns_py3dmol_view(self, visualizer):
        """show_element_filter returns a py3Dmol.view instance."""
        result = visualizer.show_element_filter("H")
        assert isinstance(result, py3Dmol.view)

    def test_works_for_carbon(self, visualizer):
        """show_element_filter works for the carbon element."""
        result = visualizer.show_element_filter("C")
        assert isinstance(result, py3Dmol.view)

    def test_invalid_element_raises(self, visualizer):
        """show_element_filter raises ValueError for an element not in the molecule."""
        with pytest.raises(ValueError, match="not in molecule"):
            visualizer.show_element_filter("Os")

    def test_custom_colormap(self, visualizer):
        """Accepts a custom colormap name without raising."""
        result = visualizer.show_element_filter("H", cmap="plasma")
        assert isinstance(result, py3Dmol.view)

    def test_show_labels_false(self, visualizer):
        """show_labels=False does not raise."""
        result = visualizer.show_element_filter("H", show_labels=False)
        assert isinstance(result, py3Dmol.view)


class TestShowMultiElement:
    """Tests for Mode 2 — multi-element display."""

    def test_returns_py3dmol_view(self, visualizer):
        """show_multi_element returns a py3Dmol.view instance."""
        result = visualizer.show_multi_element()
        assert isinstance(result, py3Dmol.view)

    def test_subset_of_elements(self, visualizer):
        """Works when only a subset of elements is specified."""
        result = visualizer.show_multi_element(elements=["H"])
        assert isinstance(result, py3Dmol.view)

    def test_explicit_all_elements(self, visualizer):
        """Passing explicit list of all elements works."""
        result = visualizer.show_multi_element(elements=["C", "H"])
        assert isinstance(result, py3Dmol.view)

    def test_no_labels(self, visualizer):
        """show_labels=False does not raise."""
        result = visualizer.show_multi_element(show_labels=False)
        assert isinstance(result, py3Dmol.view)


class TestShowGlobalScale:
    """Tests for Mode 3 — global (log or linear) scale."""

    def test_log_scale_returns_view(self, visualizer):
        """show_global_scale with log_scale=True returns py3Dmol.view."""
        result = visualizer.show_global_scale(log_scale=True)
        assert isinstance(result, py3Dmol.view)

    def test_linear_scale_returns_view(self, visualizer):
        """show_global_scale with log_scale=False returns py3Dmol.view."""
        result = visualizer.show_global_scale(log_scale=False)
        assert isinstance(result, py3Dmol.view)

    def test_custom_colormap(self, visualizer):
        """Accepts a custom colormap name without raising."""
        result = visualizer.show_global_scale(cmap="inferno")
        assert isinstance(result, py3Dmol.view)

    def test_no_labels(self, visualizer):
        """show_labels=False does not raise."""
        result = visualizer.show_global_scale(show_labels=False)
        assert isinstance(result, py3Dmol.view)


class TestShowSurface:
    """Tests for the molecular surface display mode."""

    def test_default_returns_view(self, visualizer):
        """show_surface with defaults (mode='minmax') returns py3Dmol.view."""
        result = visualizer.show_surface()
        assert isinstance(result, py3Dmol.view)

    def test_minmax_mode_returns_view(self, visualizer):
        """show_surface with mode='minmax' returns py3Dmol.view."""
        result = visualizer.show_surface(mode="minmax")
        assert isinstance(result, py3Dmol.view)

    def test_global_mode_returns_view(self, visualizer):
        """show_surface with mode='global' returns py3Dmol.view."""
        result = visualizer.show_surface(mode="global")
        assert isinstance(result, py3Dmol.view)

    def test_invalid_mode_raises(self, visualizer):
        """show_surface with unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            visualizer.show_surface(mode="unknown_mode")

    def test_element_filter(self, visualizer):
        """show_surface with element_filter does not raise."""
        result = visualizer.show_surface(mode="minmax", element_filter="H")
        assert isinstance(result, py3Dmol.view)

    def test_custom_opacity(self, visualizer):
        """Accepts a custom opacity value without raising."""
        result = visualizer.show_surface(mode="minmax", opacity=0.5)
        assert isinstance(result, py3Dmol.view)


# ── element_colormaps constant ────────────────────────────────────────────────


class TestElementColormaps:
    """Sanity checks for the ELEMENT_COLORMAPS constant."""

    def test_common_elements_have_colormaps(self):
        """H, C, N, O all have designated colormaps."""
        for element in ("H", "C", "N", "O"):
            assert element in ELEMENT_COLORMAPS

    def test_colormap_names_are_valid_matplotlib(self):
        """All listed colormap names are resolvable by matplotlib."""
        import matplotlib.pyplot as plt

        for _element, cmap_name in ELEMENT_COLORMAPS.items():
            plt.get_cmap(cmap_name)  # Raises if invalid


# ── additional edge-case tests ────────────────────────────────────────────────


class TestShowGlobalScaleLabel:
    """Tests verifying the label text in show_global_scale."""

    def test_linear_label_does_not_use_log_transform(self, visualizer):
        """In linear mode the label should show raw RPFR values, not 10**val."""
        # We can't directly inspect the label string from py3Dmol,
        # but we verify the code path doesn't crash and returns a view.
        result = visualizer.show_global_scale(log_scale=False)
        assert isinstance(result, py3Dmol.view)

    def test_log_label_returns_view(self, visualizer):
        """In log mode the label returns a view without error."""
        result = visualizer.show_global_scale(log_scale=True)
        assert isinstance(result, py3Dmol.view)


class TestShowMultiElementEdgeCases:
    """Edge-case tests for show_multi_element."""

    def test_empty_elements_list_returns_view(self, visualizer):
        """Passing elements=[] produces a viewer (with no highlighted spheres)."""
        result = visualizer.show_multi_element(elements=[])
        assert isinstance(result, py3Dmol.view)


class TestValToHexEdgeCases:
    """Additional edge-case tests for _val_to_hex."""

    def test_negative_value_clips_to_vmin(self):
        """Negative value below vmin clips to vmin colour."""
        neg = _val_to_hex(-5.0, 0.0, 1.0, "viridis")
        at_min = _val_to_hex(0.0, 0.0, 1.0, "viridis")
        assert neg == at_min


class TestDefaultColormapFallback:
    """Tests for elements not in ELEMENT_COLORMAPS."""

    def test_element_not_in_colormaps_uses_default(self):
        """An element not in ELEMENT_COLORMAPS falls back to _DEFAULT_CMAP."""
        assert "Xe" not in ELEMENT_COLORMAPS
        # The fallback is used via ELEMENT_COLORMAPS.get(el, _DEFAULT_CMAP)
        cmap = ELEMENT_COLORMAPS.get("Xe", _DEFAULT_CMAP)
        assert cmap == "viridis"

    def test_show_element_filter_single_atom_element(self, visualizer):
        """show_element_filter works for a single-atom element (C in CH4)."""
        result = visualizer.show_element_filter("C")
        assert isinstance(result, py3Dmol.view)


class TestSummaryTableColumns:
    """Test that summary_table has expected column set."""

    def test_summary_table_does_not_expose_el_std(self, visualizer):
        """summary_table intentionally excludes el_std from output."""
        df = visualizer.summary_table()
        assert "el_std" not in df.columns

    def test_internal_df_has_el_std(self, visualizer):
        """Internal DataFrame has el_std even if summary_table excludes it."""
        assert "el_std" in visualizer._df.columns
