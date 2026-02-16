"""Unit tests for IsotopeGraph."""

import pandas as pd
import pytest

from rpfr_gui.domain import IsotopeGraph


class TestIsotopeGraph:
    """Test suite for IsotopeGraph class."""

    def test_initialization(self):
        """Test graph initialization."""
        graph = IsotopeGraph(connectivity="full")

        assert graph.connectivity == "full"
        assert graph.anchor_node is None
        assert graph.graph.number_of_nodes() == 0

    def test_add_molecule(self, sample_rpfr_data):
        """Test adding a molecule to the graph."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)

        assert len(node_ids) == 5
        assert graph.graph.number_of_nodes() == 5

        # Check node attributes
        node = graph.graph.nodes[node_ids[0]]
        assert node["molecule_id"] == "mol_001"
        assert node["atom_index"] == 0
        assert node["atom_symbol"] == "C"
        assert node["rpfr"] == pytest.approx(1.12, rel=1e-6)

    def test_full_connectivity(self, sample_rpfr_data):
        """Test fully connected graph construction."""
        graph = IsotopeGraph(connectivity="full")
        graph.add_molecule("mol_001", sample_rpfr_data)
        graph.set_connectivity(mode="full")

        # Should have edges between all H atoms (4 choose 2 = 6 edges)
        # No edges to C (different element)
        assert graph.graph.number_of_edges() == 6

    def test_custom_connectivity(self, sample_rpfr_data):
        """Test custom edge definition."""
        graph = IsotopeGraph(connectivity="custom")
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)

        # Add custom edges
        custom_edges = [(node_ids[1], node_ids[2]), (node_ids[2], node_ids[3])]
        graph.set_connectivity(mode="custom", custom_edges=custom_edges)

        assert graph.graph.number_of_edges() == 2

    def test_set_anchor(self, sample_rpfr_data):
        """Test setting the anchor node."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)

        graph.set_anchor(node_ids[0])
        assert graph.anchor_node == node_ids[0]

    def test_set_anchor_invalid_node(self, sample_rpfr_data):
        """Test that setting invalid anchor raises error."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", sample_rpfr_data)

        with pytest.raises(ValueError, match="not found in graph"):
            graph.set_anchor("invalid_node_id")

    def test_get_relative_rpfr(self, sample_rpfr_data):
        """Test relative RPFR calculation.

        Isotope exchange only occurs within the same element, so relative RPFR
        is only defined when the queried node has the same element as the anchor.
        """
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)

        # Set anchor to carbon (node_ids[0], RPFR = 1.12)
        graph.set_anchor(node_ids[0])
        assert graph.anchor_element == "C"

        # Cross-element query (C anchor → H atom) must return None
        h_node = node_ids[1]  # first hydrogen
        assert graph.get_relative_rpfr(h_node) is None

        # Same-element query (C → C) returns ratio of 1.0 for the anchor itself
        c_node = node_ids[0]
        assert graph.get_relative_rpfr(c_node) == pytest.approx(1.0, rel=1e-6)

        # Same-element: add a second molecule with a known C RPFR to test ratio
        import pandas as pd

        extra = pd.DataFrame(
            [
                {"Atom_Index": 0, "Atom_Symbol": "C", "RPFR_300K": 1.12 * 2},
            ]
        )
        extra_nodes = graph.add_molecule("mol_002", extra)
        rel = graph.get_relative_rpfr(extra_nodes[0])
        assert rel == pytest.approx(2.0, rel=1e-6)

    def test_get_rpfr_dataframe(self, sample_rpfr_data):
        """Test RPFR DataFrame export."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)

        df = graph.get_rpfr_dataframe(relative=False)
        assert len(df) == 5
        assert "node_id" in df.columns
        assert "molecule_id" in df.columns
        assert "rpfr" in df.columns

        # Test relative export
        graph.set_anchor(node_ids[0])
        df_rel = graph.get_rpfr_dataframe(relative=True)
        assert "relative_rpfr" in df_rel.columns

    def test_get_subgraph_by_element(self, sample_rpfr_data):
        """Test element-specific subgraph extraction."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", sample_rpfr_data)
        graph.set_connectivity(mode="full")

        # Extract hydrogen subgraph
        h_subgraph = graph.get_subgraph_by_element("H")
        assert h_subgraph.number_of_nodes() == 4
        assert h_subgraph.number_of_edges() == 6  # 4 choose 2

        # Extract carbon subgraph
        c_subgraph = graph.get_subgraph_by_element("C")
        assert c_subgraph.number_of_nodes() == 1
        assert c_subgraph.number_of_edges() == 0

    def test_get_connected_components(self, sample_rpfr_data):
        """Test connected component identification."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", sample_rpfr_data)
        graph.set_connectivity(mode="full")

        components = graph.get_connected_components()

        # Should have 2 components: C (alone) and H-H-H-H (connected)
        assert len(components) == 2
        component_sizes = sorted([len(comp) for comp in components])
        assert component_sizes == [1, 4]

    def test_summary(self, sample_rpfr_data):
        """Test graph summary generation."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", sample_rpfr_data)
        graph.set_connectivity(mode="full")

        summary = graph.summary()

        assert summary["num_nodes"] == 5
        assert summary["num_edges"] == 6
        assert set(summary["elements"]) == {"C", "H"}
        assert summary["num_molecules"] == 1
        assert summary["connectivity_mode"] == "full"
        assert summary["anchor_set"] is False

    def test_repr(self, sample_rpfr_data):
        """Test string representation."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", sample_rpfr_data)

        repr_str = repr(graph)
        assert "IsotopeGraph" in repr_str
        assert "nodes=5" in repr_str
        assert "elements=['C', 'H']" in repr_str

class TestAnchorElement:
    """Tests for the anchor_element attribute introduced with element constraints."""

    def test_anchor_element_none_initially(self):
        """anchor_element is None before set_anchor is called."""
        graph = IsotopeGraph()
        assert graph.anchor_element is None

    def test_anchor_element_set_to_correct_symbol(self, sample_rpfr_data):
        """set_anchor stores the element symbol of the anchor node."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)

        graph.set_anchor(node_ids[0])  # C atom
        assert graph.anchor_element == "C"

    def test_anchor_element_updates_when_anchor_changes(self, sample_rpfr_data):
        """anchor_element changes when a different atom is set as anchor."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)

        graph.set_anchor(node_ids[0])  # C
        assert graph.anchor_element == "C"
        graph.set_anchor(node_ids[1])  # H
        assert graph.anchor_element == "H"

    def test_anchor_element_matches_anchor_node_symbol(self, sample_rpfr_data):
        """anchor_element always equals the atom_symbol of anchor_node."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)

        for node_id in node_ids:
            graph.set_anchor(node_id)
            expected = graph.graph.nodes[node_id]["atom_symbol"]
            assert graph.anchor_element == expected


class TestGetRelativeRPFRElementConstraint:
    """Tests for element-constrained relative RPFR calculations."""

    def test_no_anchor_returns_none(self, sample_rpfr_data):
        """get_relative_rpfr returns None when no anchor is set."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)
        assert graph.get_relative_rpfr(node_ids[0]) is None

    def test_cross_element_c_anchor_h_query_returns_none(self, sample_rpfr_data):
        """Querying H when anchor is C returns None (cross-element)."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)
        graph.set_anchor(node_ids[0])  # C anchor
        for h_node in node_ids[1:]:
            assert graph.get_relative_rpfr(h_node) is None

    def test_cross_element_h_anchor_c_query_returns_none(self, sample_rpfr_data):
        """Querying C when anchor is H returns None (cross-element)."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)
        graph.set_anchor(node_ids[1])  # H anchor
        assert graph.get_relative_rpfr(node_ids[0]) is None

    def test_anchor_node_relative_rpfr_is_one(self, sample_rpfr_data):
        """The anchor node itself always has relative RPFR of 1.0."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)
        graph.set_anchor(node_ids[0])
        assert graph.get_relative_rpfr(node_ids[0]) == pytest.approx(1.0, rel=1e-9)

    def test_same_element_ratio_correct(self, multi_element_rpfr_data):
        """Same-element ratio equals node_rpfr / anchor_rpfr."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", multi_element_rpfr_data)

        # C atoms: index 0 (RPFR=1.10) and index 1 (RPFR=1.20)
        c_anchor = node_ids[0]
        c_other = node_ids[1]
        graph.set_anchor(c_anchor)
        assert graph.get_relative_rpfr(c_other) == pytest.approx(1.20 / 1.10, rel=1e-6)

    def test_n_atoms_not_in_c_anchor_context(self, multi_element_rpfr_data):
        """N atoms return None when anchor is C (different element)."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", multi_element_rpfr_data)
        graph.set_anchor(node_ids[0])  # C anchor — N atoms are at indices 2, 3
        assert graph.get_relative_rpfr(node_ids[2]) is None
        assert graph.get_relative_rpfr(node_ids[3]) is None

    def test_zero_anchor_rpfr_raises(self):
        """Raises ValueError when anchor RPFR is zero."""
        zero_rpfr_data = pd.DataFrame(
            [
                {"Atom_Index": 0, "Atom_Symbol": "C", "RPFR_300K": 0.0},
                {"Atom_Index": 1, "Atom_Symbol": "C", "RPFR_300K": 1.2},
            ]
        )
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_zero", zero_rpfr_data)
        graph.set_anchor(node_ids[0])
        with pytest.raises(ValueError, match="Anchor RPFR is zero"):
            graph.get_relative_rpfr(node_ids[1])


class TestGetElementNormalizedRPFR:
    """Tests for the get_element_normalized_rpfr method."""

    def test_returns_dataframe(self, multi_element_rpfr_data):
        """Method returns a pandas DataFrame."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", multi_element_rpfr_data)
        assert isinstance(graph.get_element_normalized_rpfr(), pd.DataFrame)

    def test_expected_columns_present(self, multi_element_rpfr_data):
        """DataFrame contains all required columns."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", multi_element_rpfr_data)
        df = graph.get_element_normalized_rpfr(method="mean")
        for col in (
            "node_id",
            "molecule_id",
            "atom_index",
            "atom_symbol",
            "rpfr",
            "normalized_rpfr",
            "element_mean",
            "element_std",
        ):
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_matches_nodes(self, multi_element_rpfr_data):
        """DataFrame has exactly one row per node."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", multi_element_rpfr_data)
        df = graph.get_element_normalized_rpfr()
        assert len(df) == graph.graph.number_of_nodes()

    def test_mean_normalization_sum_zero_per_element(self, multi_element_rpfr_data):
        """Sum of mean-normalized values is 0 within each element."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", multi_element_rpfr_data)
        df = graph.get_element_normalized_rpfr(method="mean")
        for _, group in df.groupby("atom_symbol"):
            assert group["normalized_rpfr"].sum() == pytest.approx(0.0, abs=1e-10)

    def test_mean_normalization_formula(self, multi_element_rpfr_data):
        """mean method computes (rpfr - element_mean) / element_mean exactly."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", multi_element_rpfr_data)
        df = graph.get_element_normalized_rpfr(method="mean").reset_index(drop=True)
        expected = (df["rpfr"] - df["element_mean"]) / df["element_mean"]
        pd.testing.assert_series_equal(
            df["normalized_rpfr"], expected, check_names=False, rtol=1e-9
        )

    def test_minmax_minimum_is_zero(self, multi_element_rpfr_data):
        """Minimum minmax-normalized value per element is 0."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", multi_element_rpfr_data)
        df = graph.get_element_normalized_rpfr(method="minmax")
        for _, group in df.groupby("atom_symbol"):
            assert group["normalized_rpfr"].min() == pytest.approx(0.0, abs=1e-10)

    def test_minmax_maximum_is_one(self, multi_element_rpfr_data):
        """Maximum minmax-normalized value per element is 1."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", multi_element_rpfr_data)
        df = graph.get_element_normalized_rpfr(method="minmax")
        for _, group in df.groupby("atom_symbol"):
            assert group["normalized_rpfr"].max() == pytest.approx(1.0, abs=1e-10)

    def test_minmax_all_in_unit_interval(self, multi_element_rpfr_data):
        """All minmax-normalized values are in [0, 1]."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", multi_element_rpfr_data)
        df = graph.get_element_normalized_rpfr(method="minmax")
        assert (df["normalized_rpfr"] >= 0.0).all()
        assert (df["normalized_rpfr"] <= 1.0).all()

    def test_single_atom_minmax_is_zero(self, sample_rpfr_data):
        """Minmax value is 0 for a single-atom element (min == max)."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", sample_rpfr_data)
        df = graph.get_element_normalized_rpfr(method="minmax")
        c_rows = df[df["atom_symbol"] == "C"]
        assert c_rows["normalized_rpfr"].iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_invalid_method_raises(self, multi_element_rpfr_data):
        """Unknown normalization method raises ValueError with helpful message."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", multi_element_rpfr_data)
        with pytest.raises(ValueError, match="Unknown normalization method"):
            graph.get_element_normalized_rpfr(method="unsupported")
        with pytest.raises(ValueError, match="zscore"):
            graph.get_element_normalized_rpfr(method="zscore")

    def test_elements_normalized_independently(self, multi_element_rpfr_data):
        """Each element uses only its own atoms' statistics."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", multi_element_rpfr_data)
        df = graph.get_element_normalized_rpfr(method="mean")
        c_mean = df[df["atom_symbol"] == "C"]["element_mean"].iloc[0]
        h_mean = df[df["atom_symbol"] == "H"]["element_mean"].iloc[0]
        assert c_mean != pytest.approx(h_mean)


class TestGetRPFRDataframeRelativeElementConstraint:
    """Tests for NaN behaviour in relative mode (element constraints)."""

    def test_cross_element_atoms_are_nan(self, sample_rpfr_data):
        """With a C anchor, H atoms have NaN in the relative_rpfr column."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)
        graph.set_anchor(node_ids[0])  # C anchor

        df = graph.get_rpfr_dataframe(relative=True)
        h_rows = df[df["atom_symbol"] == "H"]
        assert h_rows["relative_rpfr"].isna().all()

    def test_anchor_element_atoms_have_valid_values(self, sample_rpfr_data):
        """Atoms of the same element as anchor have non-NaN relative_rpfr."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)
        graph.set_anchor(node_ids[0])  # C anchor

        df = graph.get_rpfr_dataframe(relative=True)
        c_rows = df[df["atom_symbol"] == "C"]
        assert c_rows["relative_rpfr"].notna().all()

    def test_anchor_node_is_one_in_dataframe(self, sample_rpfr_data):
        """The anchor node has relative_rpfr == 1.0 in the DataFrame."""
        graph = IsotopeGraph()
        node_ids = graph.add_molecule("mol_001", sample_rpfr_data)
        anchor = node_ids[0]
        graph.set_anchor(anchor)

        df = graph.get_rpfr_dataframe(relative=True)
        anchor_row = df[df["node_id"] == anchor]
        assert anchor_row["relative_rpfr"].iloc[0] == pytest.approx(1.0, rel=1e-9)

    def test_no_anchor_all_nan(self, sample_rpfr_data):
        """Without an anchor, all relative_rpfr values are NaN."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", sample_rpfr_data)
        df = graph.get_rpfr_dataframe(relative=True)
        assert df["relative_rpfr"].isna().all()


class TestIsotopeGraphEdgeCases:
    """Additional edge-case tests for IsotopeGraph."""

    def test_add_molecule_duplicate_id_overwrites_node(self, sample_rpfr_data):
        """Adding a molecule with the same ID overwrites existing nodes silently."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", sample_rpfr_data)
        original_rpfr = graph.graph.nodes["mol_001_0"]["rpfr"]

        # Modify the data and re-add with same ID
        modified = sample_rpfr_data.copy()
        modified.loc[0, "RPFR_300K"] = 999.0
        graph.add_molecule("mol_001", modified)

        # Node attributes should be overwritten
        assert graph.graph.nodes["mol_001_0"]["rpfr"] == 999.0
        assert graph.graph.nodes["mol_001_0"]["rpfr"] != original_rpfr

    def test_connectivity_not_applied_to_later_molecules(self, sample_rpfr_data):
        """Molecules added after set_connectivity are not auto-connected."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", sample_rpfr_data)
        graph.set_connectivity(mode="full")
        edges_before = graph.graph.number_of_edges()

        # Add a second molecule with H atoms
        extra = pd.DataFrame(
            [
                {"Atom_Index": 0, "Atom_Symbol": "H", "RPFR_300K": 10.0},
                {"Atom_Index": 1, "Atom_Symbol": "H", "RPFR_300K": 10.0},
            ]
        )
        graph.add_molecule("mol_002", extra)

        # Edges should NOT have increased — connectivity was set before mol_002
        assert graph.graph.number_of_edges() == edges_before

    def test_get_relative_rpfr_nonexistent_node_raises(self, sample_rpfr_data):
        """Querying a non-existent node raises KeyError."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", sample_rpfr_data)
        graph.set_anchor("mol_001_0")
        with pytest.raises(KeyError):
            graph.get_relative_rpfr("nonexistent_node")

    def test_get_rpfr_dataframe_absolute_no_nan(self, sample_rpfr_data):
        """Absolute (non-relative) DataFrame has no NaN values."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", sample_rpfr_data)
        df = graph.get_rpfr_dataframe(relative=False)
        assert df["rpfr"].notna().all()

    def test_empty_graph_summary(self):
        """Summary works on an empty graph without crashing."""
        graph = IsotopeGraph()
        summary = graph.summary()
        assert summary["num_nodes"] == 0
        assert summary["num_edges"] == 0
        assert summary["num_molecules"] == 0
        assert summary["elements"] == []

    def test_empty_graph_get_rpfr_dataframe(self):
        """get_rpfr_dataframe returns empty DataFrame for empty graph."""
        graph = IsotopeGraph()
        df = graph.get_rpfr_dataframe(relative=False)
        assert len(df) == 0

    def test_empty_graph_get_element_normalized_rpfr(self):
        """get_element_normalized_rpfr returns empty DataFrame for empty graph."""
        graph = IsotopeGraph()
        df = graph.get_element_normalized_rpfr(method="mean")
        assert len(df) == 0

    def test_get_subgraph_nonexistent_element_returns_empty(self, sample_rpfr_data):
        """Requesting a non-existent element returns an empty subgraph."""
        graph = IsotopeGraph()
        graph.add_molecule("mol_001", sample_rpfr_data)
        sub = graph.get_subgraph_by_element("Xe")
        assert sub.number_of_nodes() == 0

    def test_get_connected_components_empty_graph(self):
        """Connected components of empty graph returns empty list."""
        graph = IsotopeGraph()
        assert graph.get_connected_components() == []

    def test_add_molecule_missing_rpfr_column_raises(self):
        """Adding data without the expected RPFR column raises KeyError."""
        graph = IsotopeGraph()
        bad_data = pd.DataFrame([{"Atom_Index": 0, "Atom_Symbol": "C", "wrong_col": 1.0}])
        with pytest.raises(KeyError):
            graph.add_molecule("mol_001", bad_data)

    def test_add_molecule_missing_symbol_column_raises(self):
        """Adding data without the expected symbol column raises KeyError."""
        graph = IsotopeGraph()
        bad_data = pd.DataFrame([{"Atom_Index": 0, "wrong_col": "C", "RPFR_300K": 1.0}])
        with pytest.raises(KeyError):
            graph.add_molecule("mol_001", bad_data)

    def test_custom_edges_cross_molecule(self, sample_rpfr_data):
        """Custom edges can connect nodes from different molecules."""
        graph = IsotopeGraph()
        nodes_a = graph.add_molecule("mol_A", sample_rpfr_data)
        extra = pd.DataFrame([{"Atom_Index": 0, "Atom_Symbol": "H", "RPFR_300K": 10.0}])
        nodes_b = graph.add_molecule("mol_B", extra)
        graph.set_connectivity(mode="custom", custom_edges=[(nodes_a[1], nodes_b[0])])
        assert graph.graph.has_edge(nodes_a[1], nodes_b[0])
