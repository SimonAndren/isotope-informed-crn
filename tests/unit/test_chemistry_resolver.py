"""Unit tests for ChemistryResolver."""

import pytest

from rpfr_gui.data import ChemistryResolver


class TestChemistryResolver:
    """Test suite for ChemistryResolver class."""

    def test_canonicalize_smiles_valid(self):
        """Test canonicalization of valid SMILES."""
        resolver = ChemistryResolver.__new__(ChemistryResolver)

        # Simple molecules
        assert resolver.canonicalize_smiles("C") == "C"
        assert resolver.canonicalize_smiles("CC") == "CC"
        assert resolver.canonicalize_smiles("CO") == "CO"

        # Benzene (multiple representations)
        benzene_forms = ["c1ccccc1", "C1=CC=CC=C1"]
        canonical_benzene = resolver.canonicalize_smiles(benzene_forms[0])
        for form in benzene_forms:
            assert resolver.canonicalize_smiles(form) == canonical_benzene

    def test_canonicalize_smiles_invalid(self):
        """Test that invalid SMILES returns None."""
        resolver = ChemistryResolver.__new__(ChemistryResolver)

        assert resolver.canonicalize_smiles("invalid") is None
        assert resolver.canonicalize_smiles("C(C)(C)(C)(C)C") is None  # pentavalent carbon
        assert resolver.canonicalize_smiles("C1CCC") is None  # incomplete ring

    def test_canonicalize_inchi(self):
        """Test InChI to canonical SMILES conversion."""
        resolver = ChemistryResolver.__new__(ChemistryResolver)

        # Methane
        methane_inchi = "InChI=1S/CH4/h1H4"
        assert resolver.canonicalize_inchi(methane_inchi) == "C"

        # Ethanol
        ethanol_inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        canonical = resolver.canonicalize_inchi(ethanol_inchi)
        assert canonical is not None
        assert "C" in canonical and "O" in canonical

    def test_resolve_with_index(self, temp_index_file):
        """Test molecule ID resolution from index."""
        resolver = ChemistryResolver(temp_index_file)

        # Exact matches
        assert resolver.resolve("C", id_type="smiles") == "000001"
        assert resolver.resolve("CC", id_type="smiles") == "000002"
        assert resolver.resolve("O", id_type="smiles") == "000003"

        # Non-existent molecule
        assert resolver.resolve("CCC", id_type="smiles") is None

    def test_batch_resolve(self, temp_index_file):
        """Test batch resolution of multiple identifiers."""
        resolver = ChemistryResolver(temp_index_file)

        identifiers = ["C", "CC", "O", "CCC"]
        results = resolver.batch_resolve(identifiers, id_type="smiles")

        assert results == {
            "C": "000001",
            "CC": "000002",
            "O": "000003",
            "CCC": None,
        }

    def test_build_index(self, temp_h5_file, tmp_path):
        """Test index generation from HDF5 file."""
        output_path = tmp_path / "generated_index.csv"

        ChemistryResolver.build_index(
            temp_h5_file,
            output_path,
            smiles_dataset="SMILES",
        )

        assert output_path.exists()

        # Load and verify the generated index
        resolver = ChemistryResolver(output_path)
        assert resolver.resolve("C", id_type="smiles") == "000001"
        assert resolver.resolve("CC", id_type="smiles") == "000002"
        assert resolver.resolve("O", id_type="smiles") == "000003"

    def test_index_not_found_raises_error(self, tmp_path):
        """Test that missing index file raises appropriate error."""
        non_existent = tmp_path / "does_not_exist.csv"

        with pytest.raises(FileNotFoundError, match="Index file not found"):
            ChemistryResolver(non_existent)

    def test_build_index_numpy_array_smiles(self, temp_h5_file_numpy_smiles, tmp_path):
        """build_index handles HDF5 files where SMILES is stored as a per-atom ndarray.

        qm9s.h5 stores SMILES as an array of shape (n_atoms,) with the same
        canonical SMILES repeated once per atom.  The fix extracts only the first
        element so the resulting index contains proper canonical SMILES strings.
        """
        output_path = tmp_path / "numpy_index.csv"
        ChemistryResolver.build_index(
            temp_h5_file_numpy_smiles,
            output_path,
            smiles_dataset="SMILES",
        )

        assert output_path.exists()
        resolver = ChemistryResolver(output_path)

        # Both molecules must be resolved correctly
        assert resolver.resolve("C", id_type="smiles") == "000001"
        assert resolver.resolve("CC", id_type="smiles") == "000002"

    def test_build_index_numpy_array_smiles_parquet(self, temp_h5_file_numpy_smiles, tmp_path):
        """build_index with array SMILES writes a valid Parquet file too."""
        output_path = tmp_path / "numpy_index.parquet"
        ChemistryResolver.build_index(
            temp_h5_file_numpy_smiles,
            output_path,
            smiles_dataset="SMILES",
        )

        assert output_path.exists()
        resolver = ChemistryResolver(output_path)
        assert resolver.resolve("C", id_type="smiles") == "000001"

    def test_build_index_skips_molecules_without_smiles_dataset(self, tmp_path):
        """build_index silently skips groups that lack the SMILES dataset.

        When no valid molecules exist the output file is created but empty
        (pd.DataFrame([]).to_csv writes nothing parseable by read_csv).
        """
        import h5py
        import pandas as pd

        h5_path = tmp_path / "missing_smiles.h5"
        with h5py.File(h5_path, "w") as h5:
            grp = h5.create_group("000001")
            grp.create_dataset("Atom_Symbol", data=[b"C"])
            # Intentionally no SMILES dataset

        output_path = tmp_path / "out.csv"
        ChemistryResolver.build_index(h5_path, output_path)

        assert output_path.exists()
        # File is empty when no molecules were found
        try:
            df = pd.read_csv(output_path)
            assert len(df) == 0
        except pd.errors.EmptyDataError:
            pass  # Empty file is also valid — 0 molecules written
