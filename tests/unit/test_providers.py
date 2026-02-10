"""Unit tests for data providers."""

import pytest

from rpfr_gui.data import GNNProvider, H5Provider


class TestH5Provider:
    """Test suite for H5Provider class."""

    def test_initialization(self, temp_h5_file):
        """Test provider initialization."""
        provider = H5Provider(temp_h5_file)
        assert provider.h5_path.exists()

    def test_initialization_with_missing_file(self, tmp_path):
        """Test that missing HDF5 file raises error."""
        missing_file = tmp_path / "does_not_exist.h5"

        with pytest.raises(FileNotFoundError, match="HDF5 file not found"):
            H5Provider(missing_file)

    def test_has_molecule(self, temp_h5_file):
        """Test molecule existence checking."""
        provider = H5Provider(temp_h5_file)

        assert provider.has_molecule("000001") is True
        assert provider.has_molecule("000002") is True
        assert provider.has_molecule("000003") is True
        assert provider.has_molecule("999999") is False

    def test_get_structure(self, temp_h5_file):
        """Test SMILES retrieval."""
        provider = H5Provider(temp_h5_file)

        assert provider.get_structure("000001") == "C"
        assert provider.get_structure("000002") == "CC"
        assert provider.get_structure("000003") == "O"
        assert provider.get_structure("999999") is None

    def test_get_rpfr(self, temp_h5_file):
        """Test RPFR data retrieval."""
        provider = H5Provider(temp_h5_file)

        # Methane
        methane_data = provider.get_rpfr("000001", temperature=300.0)
        assert methane_data is not None
        assert len(methane_data) == 5
        assert list(methane_data["Atom_Symbol"]) == ["C", "H", "H", "H", "H"]
        assert methane_data["RPFR_300K"].iloc[0] == pytest.approx(1.12, rel=1e-6)

        # Water
        water_data = provider.get_rpfr("000003", temperature=300.0)
        assert water_data is not None
        assert len(water_data) == 3
        assert list(water_data["Atom_Symbol"]) == ["O", "H", "H"]

        # Non-existent molecule
        assert provider.get_rpfr("999999") is None

    def test_get_full_atom_data(self, temp_h5_file):
        """Test retrieval of multiple datasets."""
        provider = H5Provider(temp_h5_file)

        datasets = {
            "symbol": "Atom_Symbol",
            "rpfr": "RPFR_300K",
            "smiles": "SMILES",
        }

        data = provider.get_full_atom_data("000001", datasets=datasets)
        assert data is not None
        assert "symbol" in data.columns
        assert "rpfr" in data.columns
        assert "smiles" in data.columns
        assert len(data) == 5

    def test_load_batch(self, temp_h5_file):
        """Test batch loading of multiple molecules."""
        provider = H5Provider(temp_h5_file)

        molecule_ids = ["000001", "000002", "000003"]
        batch_data = provider.load_batch(
            molecule_ids,
            datasets={"symbol": "Atom_Symbol", "rpfr": "RPFR_300K"},
            show_progress=False,
        )

        assert len(batch_data) > 0
        assert "Molecule_ID" in batch_data.columns
        assert "Atom_Index" in batch_data.columns
        assert "symbol" in batch_data.columns
        assert "rpfr" in batch_data.columns

        # Check all molecules are present
        unique_mols = set(batch_data["Molecule_ID"].unique())
        assert unique_mols == {"000001", "000002", "000003"}

    def test_get_structure_numpy_array_smiles(self, temp_h5_file_numpy_smiles):
        """get_structure returns a plain string when SMILES is stored as ndarray.

        In qm9s.h5 the SMILES dataset is a per-atom array (same value repeated
        once per atom).  The fix extracts the first element so callers receive a
        regular Python string, not a list.
        """
        provider = H5Provider(temp_h5_file_numpy_smiles)

        result_ch4 = provider.get_structure("000001")
        assert isinstance(result_ch4, str), "Expected str, got list/array"
        assert result_ch4 == "C"

        result_c2h6 = provider.get_structure("000002")
        assert isinstance(result_c2h6, str)
        assert result_c2h6 == "CC"

    def test_get_rpfr_with_numpy_array_smiles_h5(self, temp_h5_file_numpy_smiles):
        """RPFR data is still accessible from an HDF5 file with array SMILES."""
        provider = H5Provider(temp_h5_file_numpy_smiles)
        df = provider.get_rpfr("000001", temperature=300.0)
        assert df is not None
        assert len(df) == 5
        assert list(df["Atom_Symbol"]) == ["C", "H", "H", "H", "H"]

    def test_has_molecule_with_numpy_array_smiles_h5(self, temp_h5_file_numpy_smiles):
        """has_molecule works regardless of SMILES storage format."""
        provider = H5Provider(temp_h5_file_numpy_smiles)
        assert provider.has_molecule("000001") is True
        assert provider.has_molecule("999999") is False


class TestGNNProvider:
    """Test suite for GNNProvider class."""

    def test_initialization(self):
        """Test GNN provider initialization."""
        provider = GNNProvider()
        assert provider.model_path is None

    def test_methods_raise_not_implemented(self):
        """Test that GNN provider methods raise NotImplementedError."""
        provider = GNNProvider()

        with pytest.raises(NotImplementedError):
            provider.get_rpfr("mol_001")

        with pytest.raises(NotImplementedError):
            provider.get_structure("mol_001")

        with pytest.raises(NotImplementedError):
            provider.has_molecule("mol_001")

        with pytest.raises(NotImplementedError):
            provider.predict("CC")
