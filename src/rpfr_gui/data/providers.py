"""Data providers for accessing RPFR data from various sources.

This module implements the Strategy pattern for data access:
- AbstractProvider: Interface defining the contract
- H5Provider: Concrete implementation for HDF5 files
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable: Iterable, *_, **__):
        return iterable


class AbstractProvider(ABC):
    """Abstract base class for RPFR data providers.

    Subclasses must implement methods to retrieve RPFR values and
    molecular structures for given molecule IDs.
    """

    @abstractmethod
    def get_rpfr(self, molecule_id: str, temperature: float = 300.0) -> pd.DataFrame | None:
        """Retrieve RPFR data for a molecule.

        Parameters
        ----------
        molecule_id : str
            Unique identifier for the molecule.
        temperature : float, optional
            Temperature in Kelvin. Default is 300.0.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns: Atom_Index, Atom_Symbol, RPFR_<temp>K
            Returns None if molecule not found.
        """
        pass

    @abstractmethod
    def get_structure(self, molecule_id: str) -> str | None:
        """Retrieve the molecular structure as SMILES.

        Parameters
        ----------
        molecule_id : str
            Unique identifier for the molecule.

        Returns
        -------
        str or None
            SMILES string, or None if not found.
        """
        pass

    @abstractmethod
    def has_molecule(self, molecule_id: str) -> bool:
        """Check if a molecule exists in the data source.

        Parameters
        ----------
        molecule_id : str
            Unique identifier for the molecule.

        Returns
        -------
        bool
            True if molecule exists, False otherwise.
        """
        pass


class H5Provider(AbstractProvider):
    """Provider for accessing RPFR data from HDF5 files.

    This provider implements lazy loading - the HDF5 file is opened
    only when data is requested, and individual molecules are read
    on-demand to avoid loading the entire >10GB dataset into memory.

    Attributes
    ----------
    h5_path : Path
        Path to the HDF5 file.
    """

    # Default dataset names for per-atom data
    DEFAULT_DATASETS: dict[str, str] = {
        "Atom_Symbol": "Atom_Symbol",
        "RPFR_300K": "RPFR_300K",
        "SMILES": "SMILES",
    }

    def __init__(self, h5_path: Path | str):
        """Initialize the HDF5 provider.

        Parameters
        ----------
        h5_path : Path or str
            Path to the HDF5 file containing RPFR data.
        """
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

    def has_molecule(self, molecule_id: str) -> bool:
        """Check if a molecule exists in the HDF5 file."""
        with h5py.File(self.h5_path, "r") as h5:
            return molecule_id in h5

    def get_structure(self, molecule_id: str) -> str | None:
        """Retrieve SMILES string for a molecule."""
        with h5py.File(self.h5_path, "r") as h5:
            if molecule_id not in h5:
                return None

            group = h5[molecule_id]
            if "SMILES" not in group:
                return None

            smiles_raw = group["SMILES"][()]
            # In qm9s.h5 SMILES is a per-atom array (same value repeated); take first element
            if isinstance(smiles_raw, np.ndarray) and smiles_raw.ndim > 0:
                smiles_raw = smiles_raw.flat[0]
            return self._decode_value(smiles_raw)

    def get_rpfr(self, molecule_id: str, temperature: float = 300.0) -> pd.DataFrame | None:
        """Retrieve RPFR data for a specific molecule at a given temperature.

        Parameters
        ----------
        molecule_id : str
            Molecule identifier (HDF5 group name).
        temperature : float, optional
            Temperature in Kelvin. Default is 300.0.

        Returns
        -------
        pd.DataFrame or None
            Per-atom RPFR data with columns: Atom_Index, Atom_Symbol, RPFR_<temp>K
        """
        temp_int = int(temperature)
        if temperature != temp_int:
            warnings.warn(
                f"Temperature {temperature} truncated to {temp_int} for HDF5 key lookup. "
                f"Pass an integer temperature to suppress this warning.",
                stacklevel=2,
            )
        rpfr_key = f"RPFR_{temp_int}K"

        with h5py.File(self.h5_path, "r") as h5:
            if molecule_id not in h5:
                return None

            group = h5[molecule_id]
            if rpfr_key not in group or "Atom_Symbol" not in group:
                return None

            # Read required datasets
            rpfr_values = group[rpfr_key][()]
            atom_symbols = group["Atom_Symbol"][()]

            # Determine number of atoms
            n_atoms = len(rpfr_values) if isinstance(rpfr_values, np.ndarray) else 1

            # Build per-atom records
            rows = []
            for idx in range(n_atoms):
                symbol = (
                    self._decode_value(atom_symbols[idx])
                    if isinstance(atom_symbols, np.ndarray)
                    else self._decode_value(atom_symbols)
                )
                rpfr = (
                    float(rpfr_values[idx])
                    if isinstance(rpfr_values, np.ndarray)
                    else float(rpfr_values)
                )

                rows.append(
                    {
                        "Atom_Index": idx,
                        "Atom_Symbol": symbol,
                        rpfr_key: rpfr,
                    }
                )

            return pd.DataFrame(rows)

    def get_full_atom_data(
        self,
        molecule_id: str,
        datasets: dict[str, str] | None = None,
    ) -> pd.DataFrame | None:
        """Retrieve multiple per-atom datasets for a molecule.

        Parameters
        ----------
        molecule_id : str
            Molecule identifier.
        datasets : dict, optional
            Mapping of column_name -> dataset_name. Defaults to DEFAULT_DATASETS.

        Returns
        -------
        pd.DataFrame or None
            Per-atom data with all requested columns.
        """
        datasets = datasets or self.DEFAULT_DATASETS

        with h5py.File(self.h5_path, "r") as h5:
            if molecule_id not in h5:
                return None

            group = h5[molecule_id]

            # Determine number of atoms (only from array datasets, not scalars)
            lengths = [
                group[dset].shape[0]
                for dset in datasets.values()
                if dset in group and group[dset].ndim > 0
            ]
            if not lengths:
                return None
            n_atoms = max(lengths)

            # Cache all datasets
            cache: dict[str, np.ndarray] = {}
            for alias, dset in datasets.items():
                if dset not in group:
                    continue
                cache[alias] = group[dset][()]

            # Build per-atom records
            rows = []
            for atom_idx in range(n_atoms):
                row = {"Atom_Index": atom_idx}
                for alias, raw in cache.items():
                    if isinstance(raw, np.ndarray):
                        if raw.ndim == 0:
                            value = self._decode_value(raw)
                        elif atom_idx < raw.shape[0]:
                            value = self._decode_value(raw[atom_idx])
                        else:
                            value = None
                    else:
                        value = self._decode_value(raw)
                    row[alias] = value
                rows.append(row)

            return pd.DataFrame(rows)

    @staticmethod
    def _decode_value(value):
        """Decode HDF5 values (bytes, arrays, JSON strings) to Python types."""
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        if isinstance(value, np.generic):
            return H5Provider._decode_value(value.item())
        if isinstance(value, np.ndarray):
            return [H5Provider._decode_value(v) for v in value.tolist()]
        if isinstance(value, list):
            return [H5Provider._decode_value(v) for v in value]
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [H5Provider._decode_value(v) for v in parsed]
            except json.JSONDecodeError:
                pass
        return value

    def load_batch(
        self,
        molecule_ids: list[str],
        datasets: dict[str, str] | None = None,
        *,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Load data for multiple molecules at once.

        Parameters
        ----------
        molecule_ids : list of str
            List of molecule IDs to load.
        datasets : dict, optional
            Mapping of column_name -> dataset_name.
        show_progress : bool, optional
            Show progress bar. Default is True.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all molecules.
        """
        datasets = datasets or self.DEFAULT_DATASETS
        all_rows = []

        iterable = tqdm(molecule_ids, desc="Loading molecules") if show_progress else molecule_ids

        with h5py.File(self.h5_path, "r") as h5:
            for mol_id in iterable:
                if mol_id not in h5:
                    continue

                group = h5[mol_id]
                lengths = [group[dset].shape[0] for dset in datasets.values() if dset in group]
                if not lengths:
                    continue
                n_atoms = max(lengths)

                cache: dict[str, np.ndarray] = {}
                for alias, dset in datasets.items():
                    if dset not in group:
                        continue
                    cache[alias] = group[dset][()]

                for atom_idx in range(n_atoms):
                    row = {"Molecule_ID": mol_id, "Atom_Index": atom_idx}
                    for alias, raw in cache.items():
                        if isinstance(raw, np.ndarray):
                            if raw.ndim == 0:
                                value = self._decode_value(raw)
                            elif atom_idx < raw.shape[0]:
                                value = self._decode_value(raw[atom_idx])
                            else:
                                value = None
                        else:
                            value = self._decode_value(raw)
                        row[alias] = value
                    all_rows.append(row)

        return pd.DataFrame(all_rows)
