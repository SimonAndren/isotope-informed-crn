"""Core data structures for the isotopologue reaction network.

Each molecular species is represented by a vector of 2^n_labeled isotopologue
concentrations, indexed by binary encoding: bit i = 1 means position i carries
the heavy isotope.

Example (3-carbon molecule, C-13 tracking):
    index 0b000 = all light (12C-12C-12C)
    index 0b001 = heavy at position 0
    index 0b101 = heavy at positions 0 and 2
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class Species:
    """A molecular species in the isotopologue network.

    Attributes:
        name: Unique identifier (e.g. 'C3H8', 'CH3').
        n_labeled: Number of atom positions that can carry the heavy isotope.
        element: Which element is tracked (e.g. 'C', 'H', 'O').
    """

    name: str
    n_labeled: int
    element: str = "C"

    @property
    def n_isotopologues(self) -> int:
        return 1 << self.n_labeled


def build_atom_map(old_bits: list[int], new_bits: list[int]) -> np.ndarray:
    """Build a permutation index array for atom rearrangement.

    Given a mapping from old bit positions to new bit positions, returns an
    index array such that ``new_conc = old_conc[result]``.

    This is a vectorized reimplementation of QIRN's MapCchain_index.

    Args:
        old_bits: Reference bit positions [n-1, ..., 0] or custom ordering.
        new_bits: Target bit positions after rearrangement.

    Returns:
        int32 array of length 2^n — a permutation of indices.
    """
    n = len(old_bits)
    nm = 1 << n
    indices = np.arange(nm, dtype=np.int32)
    new_indices = np.zeros(nm, dtype=np.int32)
    for i in range(n):
        new_indices += ((indices >> old_bits[i]) & 1).astype(np.int32) << new_bits[i]
    # new_indices[im] = new_im: maps old index → new index
    # We want result such that new_conc[new_im] = old_conc[im]
    # i.e. new_conc = old_conc[inverse_perm]
    inv = np.empty(nm, dtype=np.int32)
    inv[new_indices] = indices
    return inv


@dataclass(frozen=True, slots=True)
class AtomMap:
    """Precomputed index arrays for atom rearrangement during reactions.

    ``forward[i]`` gives the old-concentration index that maps to product index i.
    ``reverse[i]`` gives the product-concentration index that maps back to reactant index i.
    """

    forward: np.ndarray
    reverse: np.ndarray

    @staticmethod
    def from_bit_maps(
        old_bits: list[int],
        fwd_bits: list[int],
        rev_bits: list[int],
    ) -> AtomMap:
        """Construct from QIRN-style bit position lists."""
        return AtomMap(
            forward=build_atom_map(old_bits, fwd_bits),
            reverse=build_atom_map(old_bits, rev_bits),
        )


@dataclass(frozen=True)
class Reaction:
    """A reaction operating on isotopologue vectors.

    The reaction type is inferred from the number of reactants and products:
        1→1: simple transformation
        1→2: breakdown
        2→1: synthesis
        2→2: exchange

    Rate vectors include any KIE modification. Their lengths must match the
    combined isotopologue space of the reactant(s) or product(s) respectively.
    """

    name: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    k_forward: np.ndarray
    k_reverse: np.ndarray
    atom_map: AtomMap | None = None

    @property
    def reaction_type(self) -> str:
        nr, np_ = len(self.reactants), len(self.products)
        if nr == 1 and np_ == 1:
            return "simple"
        if nr == 1 and np_ == 2:
            return "breakdown"
        if nr == 2 and np_ == 1:
            return "synthesis"
        if nr == 2 and np_ == 2:
            return "exchange"
        raise ValueError(f"Unsupported reaction shape: {nr} reactants → {np_} products")


@dataclass
class Network:
    """Complete isotopologue reaction network.

    Manages the mapping between per-species isotopologue vectors and the flat
    state vector required by ODE solvers.
    """

    species: dict[str, Species]
    reactions: list[Reaction] = field(default_factory=list)

    def __post_init__(self):
        self._offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, sp in self.species.items():
            end = offset + sp.n_isotopologues
            self._offsets[name] = (offset, end)
            offset = end
        self._size = offset

    @property
    def state_size(self) -> int:
        return self._size

    def offset(self, name: str) -> tuple[int, int]:
        """Return (start, end) slice indices for a species in the flat vector."""
        return self._offsets[name]

    def pack(self, concentrations: dict[str, np.ndarray]) -> np.ndarray:
        """Pack per-species vectors into a flat ODE state vector."""
        y = np.zeros(self._size)
        for name, conc in concentrations.items():
            start, end = self._offsets[name]
            y[start:end] = conc
        return y

    def unpack(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Unpack flat ODE state vector into per-species dict."""
        result = {}
        for name in self.species:
            start, end = self._offsets[name]
            result[name] = y[start:end]
        return result
