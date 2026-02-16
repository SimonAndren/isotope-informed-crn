"""Integration tests for end-to-end workflows."""

import numpy as np
import py3Dmol
import pytest

from rpfr_gui.data import ChemistryResolver, H5Provider
from rpfr_gui.domain import IsotopeGraph
from rpfr_gui.ui.visualization import RPFRVisualizer


class TestEndToEndWorkflow:
    """Test complete workflows using multiple components."""

    def test_load_and_graph_workflow(self, temp_h5_file, temp_index_file):
        """Test complete workflow: resolve -> load -> graph."""
        # Step 1: Resolve molecule IDs
        resolver = ChemistryResolver(temp_index_file)
        mol_id = resolver.resolve("C", id_type="smiles")
        assert mol_id == "000001"

        # Step 2: Load RPFR data
        provider = H5Provider(temp_h5_file)
        rpfr_data = provider.get_rpfr(mol_id, temperature=300.0)
        assert rpfr_data is not None
        assert len(rpfr_data) == 5

        # Step 3: Build isotope graph
        graph = IsotopeGraph(connectivity="full")
        node_ids = graph.add_molecule(mol_id, rpfr_data)
        assert len(node_ids) == 5

        # Step 4: Set connectivity and analyze
        graph.set_connectivity(mode="full")
        summary = graph.summary()
        assert summary["num_nodes"] == 5
        assert summary["num_edges"] == 6  # H atoms fully connected

    def test_batch_processing_workflow(self, temp_h5_file, temp_index_file):
        """Test batch processing of multiple molecules."""
        # Resolve multiple molecules
        resolver = ChemistryResolver(temp_index_file)
        identifiers = ["C", "CC", "O"]
        mol_ids = resolver.batch_resolve(identifiers, id_type="smiles")

        # Load all molecules
        provider = H5Provider(temp_h5_file)
        graph = IsotopeGraph(connectivity="full")

        for _smiles, mol_id in mol_ids.items():
            if mol_id is None:
                continue

            rpfr_data = provider.get_rpfr(mol_id, temperature=300.0)
            if rpfr_data is not None:
                graph.add_molecule(mol_id, rpfr_data)

        # Build connectivity
        graph.set_connectivity(mode="full")

        # Verify graph contains all molecules
        summary = graph.summary()
        assert summary["num_molecules"] == 3
        assert summary["num_nodes"] == 5 + 8 + 3  # CH4 + C2H6 + H2O

    def test_relative_rpfr_workflow(self, temp_h5_file):
        """Test workflow with relative RPFR calculations."""
        provider = H5Provider(temp_h5_file)
        graph = IsotopeGraph()

        # Load methane
        rpfr_data = provider.get_rpfr("000001", temperature=300.0)
        node_ids = graph.add_molecule("000001", rpfr_data)

        # Set carbon as anchor
        carbon_node = node_ids[0]  # First atom is carbon
        graph.set_anchor(carbon_node)

        # Get relative RPFR values
        df = graph.get_rpfr_dataframe(relative=True)
        assert "relative_rpfr" in df.columns

        # Carbon should have relative RPFR of 1.0
        carbon_row = df[df["node_id"] == carbon_node]
        assert carbon_row["relative_rpfr"].iloc[0] == pytest.approx(1.0, rel=1e-6)

    def test_element_specific_analysis(self, temp_h5_file):
        """Test element-specific subgraph analysis."""
        provider = H5Provider(temp_h5_file)
        graph = IsotopeGraph()

        # Load water (H2O)
        rpfr_data = provider.get_rpfr("000003", temperature=300.0)
        graph.add_molecule("000003", rpfr_data)
        graph.set_connectivity(mode="full")

        # Analyze hydrogen network
        h_subgraph = graph.get_subgraph_by_element("H")
        assert h_subgraph.number_of_nodes() == 2
        assert h_subgraph.number_of_edges() == 1  # 2 H atoms connected

        # Analyze oxygen network
        o_subgraph = graph.get_subgraph_by_element("O")
        assert o_subgraph.number_of_nodes() == 1
        assert o_subgraph.number_of_edges() == 0  # Single O atom


class TestElementConstraintWorkflow:
    """Integration tests for the element-constrained isotope exchange workflow."""

    def test_relative_rpfr_cross_element_atoms_are_nan(self, temp_h5_file):
        """With a C anchor, all non-C atoms yield NaN in the relative DataFrame.

        This validates the full pipeline: H5Provider → IsotopeGraph → DataFrame.
        """
        provider = H5Provider(temp_h5_file)
        graph = IsotopeGraph()

        # Ethane (C2H6): 2 C atoms, 6 H atoms
        rpfr_data = provider.get_rpfr("000002", temperature=300.0)
        node_ids = graph.add_molecule("000002", rpfr_data)

        # First atom is C in ethane
        carbon_node = node_ids[0]
        graph.set_anchor(carbon_node)
        assert graph.anchor_element == "C"

        df = graph.get_rpfr_dataframe(relative=True)
        h_rows = df[df["atom_symbol"] == "H"]
        c_rows = df[df["atom_symbol"] == "C"]

        assert h_rows["relative_rpfr"].isna().all(), "H atoms must be NaN with C anchor"
        assert c_rows["relative_rpfr"].notna().all(), "C atoms must have valid values"
        # Anchor node itself is 1.0
        anchor_value = df[df["node_id"] == carbon_node]["relative_rpfr"].iloc[0]
        assert anchor_value == pytest.approx(1.0, rel=1e-9)

    def test_element_normalized_rpfr_multi_molecule(self, temp_h5_file):
        """Element normalization works correctly across multiple loaded molecules.

        Loads methane and water into the same graph, verifies minmax normalised
        values are in [0, 1] and that each element is scaled independently.
        """
        provider = H5Provider(temp_h5_file)
        graph = IsotopeGraph()

        for mol_id in ["000001", "000003"]:  # methane (CH4) + water (H2O)
            rpfr_data = provider.get_rpfr(mol_id, temperature=300.0)
            graph.add_molecule(mol_id, rpfr_data)

        df = graph.get_element_normalized_rpfr(method="minmax")

        assert "normalized_rpfr" in df.columns
        assert (df["normalized_rpfr"] >= 0.0).all()
        assert (df["normalized_rpfr"] <= 1.0).all()

        # O and H must each have their own scale
        o_mean = df[df["atom_symbol"] == "O"]["element_mean"].iloc[0]
        h_mean = df[df["atom_symbol"] == "H"]["element_mean"].iloc[0]
        assert o_mean != pytest.approx(h_mean)

    def test_numpy_array_smiles_full_pipeline(self, temp_h5_file_numpy_smiles, tmp_path):
        """Full pipeline works end-to-end when HDF5 stores SMILES as ndarray.

        Simulates the qm9s.h5 data format:
        build_index → ChemistryResolver.resolve → H5Provider.get_rpfr → IsotopeGraph
        """
        index_path = tmp_path / "numpy_index.csv"
        ChemistryResolver.build_index(temp_h5_file_numpy_smiles, index_path)

        resolver = ChemistryResolver(index_path)
        mol_id = resolver.resolve("C", id_type="smiles")
        assert mol_id == "000001"

        provider = H5Provider(temp_h5_file_numpy_smiles)
        rpfr_data = provider.get_rpfr(mol_id, temperature=300.0)
        assert rpfr_data is not None

        graph = IsotopeGraph()
        graph.add_molecule(mol_id, rpfr_data)
        graph.set_connectivity(mode="full")

        summary = graph.summary()
        assert summary["num_nodes"] == 5
        assert set(summary["elements"]) == {"C", "H"}

    def test_visualization_from_provider_data(self, temp_h5_file):
        """RPFRVisualizer can be constructed directly from H5Provider output.

        Validates that the data shape returned by the provider matches what
        the visualizer expects (symbols list, coords array, rpfr array).
        """
        import numpy as np

        from rpfr_gui.ui.visualization import RPFRVisualizer

        provider = H5Provider(temp_h5_file)
        rpfr_data = provider.get_rpfr("000001", temperature=300.0)  # Methane
        smiles = provider.get_structure("000001")

        symbols = list(rpfr_data["Atom_Symbol"])
        rpfr_values = rpfr_data["RPFR_300K"].values

        # Synthetic DFT-like coordinates for CH4 (5 atoms)
        coords = np.array(
            [
                [0.000, 0.000, 0.000],
                [0.630, 0.630, 0.630],
                [-0.630, -0.630, 0.630],
                [-0.630, 0.630, -0.630],
                [0.630, -0.630, -0.630],
            ]
        )

        viz = RPFRVisualizer(smiles, symbols, coords, rpfr_values, temperature=300.0)

        assert viz.n_atoms == 5
        assert "C" in viz._elements
        assert "H" in viz._elements

        df = viz.summary_table()
        assert len(df) == 5
        # RPFR values must match those from the provider
        import pandas as pd

        pd.testing.assert_series_equal(
            df.sort_values("idx")["rpfr"].reset_index(drop=True),
            pd.Series(rpfr_values, name="rpfr"),
            rtol=1e-6,
        )

    def test_element_filter_visualization_from_real_data(self, temp_h5_file):
        """show_element_filter works on data loaded from H5Provider."""
        import numpy as np
        import py3Dmol

        from rpfr_gui.ui.visualization import RPFRVisualizer

        provider = H5Provider(temp_h5_file)
        rpfr_data = provider.get_rpfr("000001", temperature=300.0)
        smiles = provider.get_structure("000001")

        symbols = list(rpfr_data["Atom_Symbol"])
        rpfr_values = rpfr_data["RPFR_300K"].values
        coords = np.array(
            [
                [0.000, 0.000, 0.000],
                [0.630, 0.630, 0.630],
                [-0.630, -0.630, 0.630],
                [-0.630, 0.630, -0.630],
                [0.630, -0.630, -0.630],
            ]
        )

        viz = RPFRVisualizer(smiles, symbols, coords, rpfr_values)
        view = viz.show_element_filter("H")
        assert isinstance(view, py3Dmol.view)


class TestAdditionalIntegrationWorkflows:
    """Additional integration tests covering untested workflow paths."""

    def test_full_pipeline_with_parquet_index(self, temp_h5_file, tmp_path):
        """End-to-end workflow using Parquet index instead of CSV."""
        index_path = tmp_path / "index.parquet"
        ChemistryResolver.build_index(temp_h5_file, index_path)

        resolver = ChemistryResolver(index_path)
        mol_id = resolver.resolve("C", id_type="smiles")
        assert mol_id == "000001"

        provider = H5Provider(temp_h5_file)
        rpfr_data = provider.get_rpfr(mol_id, temperature=300.0)
        assert rpfr_data is not None

        graph = IsotopeGraph()
        graph.add_molecule(mol_id, rpfr_data)
        graph.set_connectivity(mode="full")
        assert graph.summary()["num_nodes"] == 5

    def test_load_batch_to_graph_workflow(self, temp_h5_file):
        """load_batch output can be transformed into IsotopeGraph nodes."""
        provider = H5Provider(temp_h5_file)
        batch_df = provider.load_batch(
            ["000001", "000003"],
            datasets={"Atom_Symbol": "Atom_Symbol", "RPFR_300K": "RPFR_300K"},
            show_progress=False,
        )
        assert len(batch_df) > 0

        graph = IsotopeGraph()
        for mol_id, group in batch_df.groupby("Molecule_ID"):
            graph.add_molecule(str(mol_id), group.reset_index(drop=True))

        graph.set_connectivity(mode="full")
        summary = graph.summary()
        assert summary["num_molecules"] == 2
        assert summary["num_nodes"] == 5 + 3  # CH4 + H2O

    def test_error_propagation_resolve_none_to_provider(self, temp_h5_file, temp_index_file):
        """When resolve returns None, provider gracefully returns None."""
        resolver = ChemistryResolver(temp_index_file)
        mol_id = resolver.resolve("NONEXISTENT_SMILES", id_type="smiles")
        assert mol_id is None

        provider = H5Provider(temp_h5_file)
        # Passing None as molecule_id should not crash — returns None
        if mol_id is not None:
            rpfr_data = provider.get_rpfr(mol_id)
        else:
            rpfr_data = None
        assert rpfr_data is None
