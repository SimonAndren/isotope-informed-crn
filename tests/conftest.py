"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return {
        "methane": "C",
        "ethane": "CC",
        "methanol": "CO",
        "water": "O",
        "benzene": "c1ccccc1",
    }


@pytest.fixture
def sample_rpfr_data():
    """Sample RPFR data for testing."""
    return pd.DataFrame(
        [
            {"Atom_Index": 0, "Atom_Symbol": "C", "RPFR_300K": 1.12},
            {"Atom_Index": 1, "Atom_Symbol": "H", "RPFR_300K": 11.5},
            {"Atom_Index": 2, "Atom_Symbol": "H", "RPFR_300K": 11.5},
            {"Atom_Index": 3, "Atom_Symbol": "H", "RPFR_300K": 11.5},
            {"Atom_Index": 4, "Atom_Symbol": "H", "RPFR_300K": 11.5},
        ]
    )


@pytest.fixture
def temp_h5_file(tmp_path):
    """Create a temporary HDF5 file with sample data for testing."""
    h5_path = tmp_path / "test_rpfr.h5"

    with h5py.File(h5_path, "w") as h5:
        # Add methane (CH4)
        methane = h5.create_group("000001")
        methane.create_dataset("SMILES", data=b"C")
        methane.create_dataset("Atom_Symbol", data=np.array([b"C", b"H", b"H", b"H", b"H"]))
        methane.create_dataset("RPFR_300K", data=np.array([1.12, 11.5, 11.5, 11.5, 11.5]))

        # Add ethane (C2H6)
        ethane = h5.create_group("000002")
        ethane.create_dataset("SMILES", data=b"CC")
        ethane.create_dataset(
            "Atom_Symbol", data=np.array([b"C", b"C", b"H", b"H", b"H", b"H", b"H", b"H"])
        )
        ethane.create_dataset(
            "RPFR_300K", data=np.array([1.15, 1.15, 11.3, 11.3, 11.3, 11.3, 11.3, 11.3])
        )

        # Add water (H2O)
        water = h5.create_group("000003")
        water.create_dataset("SMILES", data=b"O")
        water.create_dataset("Atom_Symbol", data=np.array([b"O", b"H", b"H"]))
        water.create_dataset("RPFR_300K", data=np.array([1.03, 12.1, 12.1]))

    return h5_path


@pytest.fixture
def temp_index_file(tmp_path, sample_smiles):
    """Create a temporary index CSV file."""
    index_path = tmp_path / "index.csv"

    records = [
        {
            "Molecule_ID": "000001",
            "Original_SMILES": "C",
            "Canonical_SMILES": "C",
        },
        {
            "Molecule_ID": "000002",
            "Original_SMILES": "CC",
            "Canonical_SMILES": "CC",
        },
        {
            "Molecule_ID": "000003",
            "Original_SMILES": "O",
            "Canonical_SMILES": "O",
        },
    ]

    pd.DataFrame(records).to_csv(index_path, index=False)
    return index_path


@pytest.fixture
def temp_h5_file_numpy_smiles(tmp_path):
    """HDF5 fixture where SMILES is stored as a per-atom numpy array (like qm9s.h5)."""
    h5_path = tmp_path / "test_numpy_smiles.h5"

    with h5py.File(h5_path, "w") as h5:
        # Methane (CH4) — SMILES repeated once per atom
        methane = h5.create_group("000001")
        methane.create_dataset(
            "SMILES",
            data=np.array([b"C", b"C", b"C", b"C", b"C"], dtype=object),
        )
        methane.create_dataset("Atom_Symbol", data=np.array([b"C", b"H", b"H", b"H", b"H"]))
        methane.create_dataset("RPFR_300K", data=np.array([1.12, 11.5, 11.5, 11.5, 11.5]))

        # Ethane (C2H6) — SMILES repeated once per atom
        ethane = h5.create_group("000002")
        ethane.create_dataset(
            "SMILES",
            data=np.array([b"CC"] * 8, dtype=object),
        )
        ethane.create_dataset(
            "Atom_Symbol",
            data=np.array([b"C", b"C", b"H", b"H", b"H", b"H", b"H", b"H"]),
        )
        ethane.create_dataset(
            "RPFR_300K",
            data=np.array([1.15, 1.15, 11.3, 11.3, 11.3, 11.3, 11.3, 11.3]),
        )

    return h5_path


@pytest.fixture
def multi_element_rpfr_data():
    """RPFR data with multiple atoms per element for normalization tests (C, N, H)."""
    return pd.DataFrame(
        [
            {"Atom_Index": 0, "Atom_Symbol": "C", "RPFR_300K": 1.10},
            {"Atom_Index": 1, "Atom_Symbol": "C", "RPFR_300K": 1.20},
            {"Atom_Index": 2, "Atom_Symbol": "N", "RPFR_300K": 1.09},
            {"Atom_Index": 3, "Atom_Symbol": "N", "RPFR_300K": 1.11},
            {"Atom_Index": 4, "Atom_Symbol": "H", "RPFR_300K": 11.0},
            {"Atom_Index": 5, "Atom_Symbol": "H", "RPFR_300K": 12.0},
            {"Atom_Index": 6, "Atom_Symbol": "H", "RPFR_300K": 13.0},
        ]
    )


@pytest.fixture
def sample_visualizer_data():
    """Input data for RPFRVisualizer tests — methane (CH4), 5 atoms.

    RPFR values are deliberately non-uniform so element statistics are meaningful.
    Coordinates are approximate DFT-like positions in Angstroms.
    """
    symbols = ["C", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.000, 0.000, 0.000],  # C
            [0.630, 0.630, 0.630],  # H1
            [-0.630, -0.630, 0.630],  # H2
            [-0.630, 0.630, -0.630],  # H3
            [0.630, -0.630, -0.630],  # H4
        ]
    )
    rpfr = np.array([1.12, 11.4, 11.5, 11.6, 11.7])
    return {
        "smiles": "C",
        "symbols": symbols,
        "coords": coords,
        "rpfr": rpfr,
        "temperature": 300.0,
    }
