"""3D RPFR visualization using RDKit and py3Dmol.

Renders molecules with per-atom RPFR values as surface/sphere color overlays.
Supports three display modes for handling the cross-element scale problem:

1. element_filter  – Show one element at a time (scientifically strictest)
2. multi_element   – All elements displayed, independent colormap per element
3. global_scale    – All RPFR values on one shared (log) scale
"""

from __future__ import annotations

from collections.abc import Sequence

# matplotlib colormaps (for color lookup)
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# py3Dmol
import py3Dmol

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# ── colour helpers ────────────────────────────────────────────────────────────

# One perceptually-uniform colormap per element (colour-blind safe)
ELEMENT_COLORMAPS: dict[str, str] = {
    "H": "Blues",
    "C": "Greens",
    "N": "Purples",
    "O": "Oranges",
    "S": "YlOrRd",
    "P": "RdPu",
    "F": "cool",
    "Cl": "BuGn",
    "Br": "autumn",
    "I": "hot",
}
_DEFAULT_CMAP = "viridis"

# CPK-ish element colours used for atom labels / fallback spheres
CPK_COLORS: dict[str, str] = {
    "H": "#FFFFFF",
    "C": "#404040",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "S": "#FFFF30",
    "P": "#FF8000",
    "F": "#90E050",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "I": "#940094",
}


def _val_to_hex(value: float, vmin: float, vmax: float, cmap_name: str) -> str:
    """Map a scalar value in [vmin, vmax] to a hex colour string."""
    cmap = plt.get_cmap(cmap_name)
    if vmax == vmin:
        norm_val = 0.5
    else:
        norm_val = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
    r, g, b, _ = cmap(norm_val)
    return mcolors.to_hex((r, g, b))


# ── molecule builder ──────────────────────────────────────────────────────────


def build_rdkit_mol(
    smiles: str,
    atom_symbols: list[str],
    coords: np.ndarray,
) -> Chem.Mol:
    """Create an RDKit Mol with DFT-optimised coordinates from the HDF5 file.

    Parameters
    ----------
    smiles : str
        Canonical SMILES string for the molecule.
    atom_symbols : list of str
        Element symbol for each atom (ordered as in HDF5).
    coords : np.ndarray
        Shape (n_atoms, 3) — x, y, z in Angströms from DFT optimisation.

    Returns
    -------
    rdkit.Chem.Mol
        Molecule with a single embedded 3D conformer.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    # Add explicit Hs so the atom ordering matches the HDF5 file
    mol = Chem.AddHs(mol)

    # Embed a conformer using distance geometry (placeholder positions)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    conf = mol.GetConformer()
    n_rdkit = mol.GetNumAtoms()

    if len(atom_symbols) != n_rdkit:
        raise ValueError(
            f"Atom count mismatch: HDF5 has {len(atom_symbols)} atoms, "
            f"RDKit has {n_rdkit} atoms. "
            "Check that the SMILES represents the same molecule."
        )

    # Replace generated coordinates with the DFT-optimised ones
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))

    return mol


def mol_to_xyz_block(mol: Chem.Mol) -> str:
    """Serialise a molecule to XYZ format for py3Dmol."""
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    lines = [str(n), ""]
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        lines.append(f"{sym:2s}  {pos.x:12.6f}  {pos.y:12.6f}  {pos.z:12.6f}")
    return "\n".join(lines)


# ── main visualiser ───────────────────────────────────────────────────────────


class RPFRVisualizer:
    """Interactive 3D visualisation of per-atom RPFR values.

    Parameters
    ----------
    smiles : str
        Canonical SMILES for the molecule.
    atom_symbols : list of str
        Element for each atom (same order as coords / rpfr_values).
    coords : np.ndarray
        DFT-optimised 3D coordinates, shape (n_atoms, 3), in Angströms.
    rpfr_values : np.ndarray
        RPFR value for each atom, shape (n_atoms,).
    temperature : float, optional
        Temperature label for the title. Default 300.
    """

    def __init__(
        self,
        smiles: str,
        atom_symbols: list[str],
        coords: np.ndarray,
        rpfr_values: np.ndarray,
        temperature: float = 300.0,
    ):
        self.smiles = smiles
        self.symbols = list(atom_symbols)
        self.coords = np.asarray(coords)
        self.rpfr = np.asarray(rpfr_values, dtype=float)
        self.temperature = temperature
        self.n_atoms = len(self.symbols)

        # Build RDKit mol (sets up atom ordering + DFT coords)
        self.mol = build_rdkit_mol(smiles, self.symbols, self.coords)
        self.xyz_block = mol_to_xyz_block(self.mol)

        # Per-element statistics (used by multiple modes)
        self._df = pd.DataFrame(
            {
                "idx": np.arange(self.n_atoms),
                "symbol": self.symbols,
                "rpfr": self.rpfr,
                "x": self.coords[:, 0],
                "y": self.coords[:, 1],
                "z": self.coords[:, 2],
            }
        )
        stats = self._df.groupby("symbol")["rpfr"].agg(["mean", "std", "min", "max"])
        self._df = self._df.join(
            stats.rename(
                columns={"mean": "el_mean", "std": "el_std", "min": "el_min", "max": "el_max"}
            ),
            on="symbol",
        )
        self._df["minmax"] = (self._df["rpfr"] - self._df["el_min"]) / (
            self._df["el_max"] - self._df["el_min"]
        ).replace(0, 1.0)
        self._elements = sorted(self._df["symbol"].unique())

    # ── private helpers ───────────────────────────────────────────────────────

    def _base_viewer(self, width: int = 700, height: int = 500) -> py3Dmol.view:
        """Create a py3Dmol viewer pre-loaded with the molecule."""
        view = py3Dmol.view(width=width, height=height)
        view.addModel(self.xyz_block, "xyz")
        return view

    def _add_rpfr_spheres(
        self,
        view: py3Dmol.view,
        colors: dict[int, str],
        values: dict[int, float],
        radius_scale: float = 0.35,
        show_labels: bool = True,
    ):
        """Add one coloured sphere per atom, sized and labelled by RPFR."""
        for idx, hex_color in colors.items():
            pos = self.coords[idx]
            sym = self.symbols[idx]
            rpfr_val = values[idx]

            view.addSphere(
                {
                    "center": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                    "radius": radius_scale,
                    "color": hex_color,
                    "opacity": 0.85,
                }
            )
            if show_labels:
                view.addLabel(
                    f"{sym}{idx}\n{rpfr_val:.3f}",
                    {
                        "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                        "fontSize": 10,
                        "fontColor": "white",
                        "backgroundColor": "black",
                        "backgroundOpacity": 0.5,
                        "alignment": "center",
                    },
                )

    def _add_bonds(self, view: py3Dmol.view):
        """Draw bond sticks between bonded atoms."""
        view.setStyle({"model": 0}, {"stick": {"radius": 0.08, "color": "grey"}})

    # ── public display modes ──────────────────────────────────────────────────

    def show_element_filter(
        self,
        element: str,
        cmap: str | None = None,
        width: int = 700,
        height: int = 500,
        show_labels: bool = True,
    ) -> py3Dmol.view:
        """Mode 1 — Show only one element, coloured by its RPFR.

        Other atoms are shown as faint grey sticks for structural context.
        RPFR scale is min-max within the selected element.

        Parameters
        ----------
        element : str
            Atomic symbol to highlight (e.g. "H", "C", "N", "O").
        cmap : str, optional
            Matplotlib colormap name. Defaults to element-specific map.
        """
        if element not in self._elements:
            raise ValueError(f"Element {element!r} not in molecule. Available: {self._elements}")

        cmap_name = cmap or ELEMENT_COLORMAPS.get(element, _DEFAULT_CMAP)
        sub = self._df[self._df["symbol"] == element]
        vmin, vmax = sub["rpfr"].min(), sub["rpfr"].max()

        view = self._base_viewer(width, height)
        # Grey sticks for all atoms as backbone
        view.setStyle({}, {"stick": {"radius": 0.06, "color": "lightgrey"}})

        colors = {}
        values = {}
        for _, row in sub.iterrows():
            colors[int(row["idx"])] = _val_to_hex(row["rpfr"], vmin, vmax, cmap_name)
            values[int(row["idx"])] = row["rpfr"]

        self._add_rpfr_spheres(view, colors, values, radius_scale=0.30, show_labels=show_labels)

        view.addLabel(
            f"{element} atoms  |  T={self.temperature:.0f}K  |  RPFR [{vmin:.4f}, {vmax:.4f}]",
            {
                "position": {"x": 0, "y": 0, "z": 0},
                "fontSize": 12,
                "fontColor": "black",
                "backgroundColor": "white",
                "backgroundOpacity": 0.7,
                "screenOffset": {"x": 0, "y": -200},
            },
        )
        view.zoomTo()
        return view

    def show_multi_element(
        self,
        elements: Sequence[str] | None = None,
        width: int = 800,
        height: int = 550,
        show_labels: bool = True,
    ) -> py3Dmol.view:
        """Mode 2 — All elements visible, each with its own colour scale.

        Each element gets an independent min-max normalisation and its own
        matplotlib colormap (see ELEMENT_COLORMAPS). The absolute RPFR
        values are not cross-comparable across elements, but site-specific
        variations within each element are clearly visible.
        """
        elements = list(elements) if elements else self._elements
        view = self._base_viewer(width, height)
        view.setStyle({}, {"stick": {"radius": 0.06, "color": "lightgrey"}})

        colors: dict[int, str] = {}
        values: dict[int, float] = {}

        for el in elements:
            sub = self._df[self._df["symbol"] == el]
            cmap_name = ELEMENT_COLORMAPS.get(el, _DEFAULT_CMAP)
            vmin, vmax = sub["rpfr"].min(), sub["rpfr"].max()
            for _, row in sub.iterrows():
                colors[int(row["idx"])] = _val_to_hex(row["rpfr"], vmin, vmax, cmap_name)
                values[int(row["idx"])] = row["rpfr"]

        self._add_rpfr_spheres(view, colors, values, radius_scale=0.28, show_labels=show_labels)

        # Legend labels in upper corner
        for i, el in enumerate(elements):
            cmap_name = ELEMENT_COLORMAPS.get(el, _DEFAULT_CMAP)
            sub = self._df[self._df["symbol"] == el]
            vmin, vmax = sub["rpfr"].min(), sub["rpfr"].max()
            view.addLabel(
                f"{el}: [{vmin:.3f} – {vmax:.3f}]",
                {
                    "fontSize": 11,
                    "fontColor": "black",
                    "backgroundColor": "white",
                    "backgroundOpacity": 0.65,
                    "screenOffset": {"x": -280, "y": -170 + i * 18},
                },
            )

        view.zoomTo()
        return view

    def show_global_scale(
        self,
        log_scale: bool = True,
        cmap: str = "viridis",
        width: int = 700,
        height: int = 500,
        show_labels: bool = True,
    ) -> py3Dmol.view:
        """Mode 3 — All elements on one shared (optionally log) colour scale.

        Useful for a raw overview of the data. Because H RPFR (~10-15) is
        ~10x larger than C/N/O (~1.1-1.2), a log scale is strongly recommended
        to see variation within the heavy atoms.

        Parameters
        ----------
        log_scale : bool
            Apply log10 transform before colouring. Default True.
        cmap : str
            Matplotlib colormap. Default "viridis".
        """
        view = self._base_viewer(width, height)
        view.setStyle({}, {"stick": {"radius": 0.06, "color": "lightgrey"}})

        rpfr_vals = self._df["rpfr"].values
        display_vals = np.log10(rpfr_vals) if log_scale else rpfr_vals
        vmin, vmax = display_vals.min(), display_vals.max()

        colors: dict[int, str] = {}
        values: dict[int, float] = {}
        for _, row in self._df.iterrows():
            val = np.log10(row["rpfr"]) if log_scale else row["rpfr"]
            colors[int(row["idx"])] = _val_to_hex(val, vmin, vmax, cmap)
            values[int(row["idx"])] = row["rpfr"]

        self._add_rpfr_spheres(view, colors, values, radius_scale=0.28, show_labels=show_labels)

        scale_label = "log₁₀(RPFR)" if log_scale else "RPFR"
        view.addLabel(
            f"{scale_label}  [{10**vmin:.3f} – {10**vmax:.3f}]  T={self.temperature:.0f}K",
            {
                "fontSize": 12,
                "fontColor": "black",
                "backgroundColor": "white",
                "backgroundOpacity": 0.7,
                "screenOffset": {"x": 0, "y": -220},
            },
        )
        view.zoomTo()
        return view

    def show_surface(
        self,
        mode: str = "minmax",
        element_filter: str | None = None,
        opacity: float = 0.85,
        cmap: str = "viridis",
        width: int = 800,
        height: int = 550,
    ) -> py3Dmol.view:
        """Molecular surface (SAS) coloured by the nearest atom's RPFR.

        This produces an electron-cloud-like surface where the colour at
        each surface point reflects the RPFR of the nearest heavy atom.

        Parameters
        ----------
        mode : str
            "minmax"  : per-element min-max colouring (default)
            "global"  : log10-normalised global scale
        element_filter : str, optional
            If given, only that element's atoms contribute to the surface colour.
        opacity : float
            Surface transparency (0=transparent, 1=opaque). Default 0.85.
        cmap : str
            Matplotlib colormap name.
        """
        view = self._base_viewer(width, height)
        # Backbone sticks
        view.setStyle({}, {"stick": {"radius": 0.08, "color": "grey"}})

        # Compute colour per atom
        if mode == "minmax":
            norm_vals = self._df["minmax"].values
        elif mode == "global":
            log_rpfr = np.log10(self._df["rpfr"].values)
            norm_vals = (log_rpfr - log_rpfr.min()) / (log_rpfr.max() - log_rpfr.min() + 1e-9)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'minmax' or 'global'.")

        cm = plt.get_cmap(cmap)

        # Build a py3Dmol surface coloured by property
        # py3Dmol's addSurface accepts a 'colorscheme' dict with atom-mapped colours
        prop_map = {}
        for i, (_, row) in enumerate(self._df.iterrows()):
            if element_filter and row["symbol"] != element_filter:
                continue
            r, g, b, _ = cm(float(norm_vals[i]))
            prop_map[int(row["idx"])] = mcolors.to_hex((r, g, b))

        # Fallback colour for atoms not in the map
        fallback = "#aaaaaa"

        # Build atom colour list in index order for py3Dmol's colorscheme
        color_list = [prop_map.get(i, fallback) for i in range(self.n_atoms)]

        # py3Dmol surface with per-atom colour
        view.addSurface(
            py3Dmol.SAS,
            {
                "opacity": opacity,
                "colorscheme": {"prop": "index", "gradient": "roygb"},
            },
        )
        # Override with atom-specific colours via sphere overlay
        for i, hex_c in enumerate(color_list):
            if element_filter and self.symbols[i] != element_filter:
                continue
            pos = self.coords[i]
            view.addSphere(
                {
                    "center": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                    "radius": 0.25,
                    "color": hex_c,
                    "opacity": 1.0,
                }
            )

        view.zoomTo()
        return view

    def summary_table(self) -> pd.DataFrame:
        """Return a tidy summary table of all atoms with their RPFR and normalised values."""
        return self._df[["idx", "symbol", "rpfr", "el_mean", "el_min", "el_max", "minmax"]].copy()
