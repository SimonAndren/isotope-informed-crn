"""Chemistry resolver for mapping chemical identifiers to database indices.

This module handles canonicalization of SMILES/InChI strings and provides
fast lookup of molecule indices in the HDF5 dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


class ChemistryResolver:
    """Maps chemical identifiers to HDF5 row indices.

    This class handles:
    - Canonicalization of SMILES/InChI strings using RDKit
    - Generation and caching of lookup indices (Parquet/CSV)
    - Fast molecule ID resolution

    Attributes
    ----------
    index_path : Path
        Path to the cached index file.
    lookup_table : pd.DataFrame
        In-memory lookup table mapping canonical SMILES to molecule IDs.
    """

    def __init__(
        self,
        index_path: Path | str,
        *,
        auto_generate: bool = True,
    ):
        """Initialize the chemistry resolver.

        Parameters
        ----------
        index_path : Path or str
            Path to the index file (Parquet or CSV). If it doesn't exist and
            auto_generate is True, it will be created from the HDF5 file.
        auto_generate : bool, optional
            If True (default) and the index file doesn't exist, raise
            FileNotFoundError. If False, silently allow a missing index
            (useful when the caller plans to call build_index() later).
        """
        self.index_path = Path(index_path)
        self.lookup_table: pd.DataFrame | None = None

        if self.index_path.exists():
            self._load_index()
        elif not auto_generate:
            # Silently allow missing index when auto_generate is False;
            # the caller is expected to call build_index() later.
            pass
        else:
            raise FileNotFoundError(
                f"Index file not found at {self.index_path}. "
                "Please generate it using build_index() or provide a valid path."
            )

    def _load_index(self) -> None:
        """Load the lookup index from disk."""
        if self.index_path.suffix == ".parquet":
            self.lookup_table = pd.read_parquet(self.index_path)
        elif self.index_path.suffix == ".csv":
            # Read CSV with Molecule_ID as string type to preserve leading zeros
            self.lookup_table = pd.read_csv(self.index_path, dtype={"Molecule_ID": str})
        else:
            raise ValueError(f"Unsupported index format: {self.index_path.suffix}")

        # Ensure required columns exist
        required = {"Canonical_SMILES", "Molecule_ID"}
        if not required.issubset(self.lookup_table.columns):
            raise ValueError(
                f"Index must contain columns {required}, found {set(self.lookup_table.columns)}"
            )

    def canonicalize_smiles(self, smiles: str) -> str | None:
        """Convert a SMILES string to canonical form.

        Parameters
        ----------
        smiles : str
            Input SMILES string.

        Returns
        -------
        str or None
            Canonical SMILES string, or None if parsing fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def canonicalize_inchi(self, inchi: str) -> str | None:
        """Convert an InChI string to canonical SMILES.

        Parameters
        ----------
        inchi : str
            Input InChI string.

        Returns
        -------
        str or None
            Canonical SMILES string, or None if parsing fails.
        """
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def resolve(self, identifier: str, *, id_type: str = "smiles") -> str | None:
        """Resolve a chemical identifier to a molecule ID.

        Parameters
        ----------
        identifier : str
            Chemical identifier (SMILES or InChI).
        id_type : str, optional
            Type of identifier: "smiles" or "inchi". Default is "smiles".

        Returns
        -------
        str or None
            Molecule ID if found in the database, None otherwise.
        """
        if self.lookup_table is None:
            raise RuntimeError("Index not loaded. Call _load_index() first.")

        # Canonicalize the input
        if id_type.lower() == "smiles":
            canonical = self.canonicalize_smiles(identifier)
        elif id_type.lower() == "inchi":
            canonical = self.canonicalize_inchi(identifier)
        else:
            raise ValueError(f"Unsupported id_type: {id_type}")

        if canonical is None:
            return None

        # Lookup in the index
        matches = self.lookup_table[self.lookup_table["Canonical_SMILES"] == canonical]
        if matches.empty:
            return None

        return matches.iloc[0]["Molecule_ID"]

    def batch_resolve(
        self, identifiers: list[str], *, id_type: str = "smiles"
    ) -> dict[str, str | None]:
        """Resolve multiple identifiers at once.

        Parameters
        ----------
        identifiers : list of str
            List of chemical identifiers.
        id_type : str, optional
            Type of identifier: "smiles" or "inchi". Default is "smiles".

        Returns
        -------
        dict
            Mapping from input identifier to molecule ID (or None if not found).
        """
        return {ident: self.resolve(ident, id_type=id_type) for ident in identifiers}

    @staticmethod
    def build_index(
        h5_path: Path | str,
        output_path: Path | str,
        *,
        smiles_dataset: str = "SMILES",
        limit: int | None = None,
    ) -> None:
        """Generate a lookup index from an HDF5 file.

        Parameters
        ----------
        h5_path : Path or str
            Path to the HDF5 file.
        output_path : Path or str
            Path where the index will be saved (Parquet or CSV).
        smiles_dataset : str, optional
            Name of the SMILES dataset in the HDF5 file. Default is "SMILES".
        limit : int, optional
            Maximum number of molecules to process (for testing). Default is None.
        """
        import h5py

        h5_path = Path(h5_path)
        output_path = Path(output_path)

        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        records = []
        with h5py.File(h5_path, "r") as h5:
            molecule_ids = sorted(h5.keys())[:limit] if limit else sorted(h5.keys())

            for mol_id in molecule_ids:
                group = h5[mol_id]
                if smiles_dataset not in group:
                    continue

                # Read SMILES (may be scalar bytes, scalar str, or per-atom array)
                smiles_raw = group[smiles_dataset][()]
                if isinstance(smiles_raw, bytes):
                    smiles_str = smiles_raw.decode("utf-8")
                elif isinstance(smiles_raw, str):
                    smiles_str = smiles_raw
                elif isinstance(smiles_raw, np.ndarray) and smiles_raw.size > 0:
                    # Per-atom array (e.g. qm9s.h5): same SMILES repeated per atom
                    first = smiles_raw.flat[0]
                    smiles_str = first.decode("utf-8") if isinstance(first, bytes) else str(first)
                else:
                    continue

                # Canonicalize
                mol = Chem.MolFromSmiles(smiles_str)
                if mol is None:
                    continue
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

                records.append(
                    {
                        "Molecule_ID": mol_id,
                        "Original_SMILES": smiles_str,
                        "Canonical_SMILES": canonical_smiles,
                    }
                )

        # Save the index
        df = pd.DataFrame(records)
        if output_path.suffix == ".parquet":
            df.to_parquet(output_path, index=False)
        elif output_path.suffix == ".csv":
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

        print(f"Index generated: {len(df)} entries saved to {output_path}")
