"""Isotope analysis: delta notation, site-specific ratios, mass spectra.

All functions operate on isotopologue concentration vectors (length 2^n)
using the binary index convention: bit i = 1 means position i carries the
heavy isotope.
"""

from __future__ import annotations

import numpy as np

# Standard reference ratios
VPDB_RATIO = 0.01118  # 13C/12C Vienna Pee Dee Belemnite
VSMOW_D_RATIO = 155.76e-6  # D/H Vienna Standard Mean Ocean Water
VSMOW_18O_RATIO = 2005.2e-6  # 18O/16O

REFERENCE_RATIOS: dict[str, float] = {
    "C": VPDB_RATIO,
    "H": VSMOW_D_RATIO,
    "O": VSMOW_18O_RATIO,
}


def total_ratio(conc: np.ndarray) -> float:
    """Compound-specific isotope ratio (heavy/light) from isotopologue vector.

    Sums the weighted heavy and light atom counts across all isotopologues.

    Args:
        conc: Isotopologue concentration vector of length 2^n.

    Returns:
        The bulk isotope ratio (heavy / light) for the molecule.
    """
    nm = len(conc)
    n = int(np.log2(nm))
    total_heavy = 0.0
    total_light = 0.0
    for im in range(nm):
        n_heavy = bin(im).count("1")
        n_light = n - n_heavy
        total_heavy += conc[im] * n_heavy
        total_light += conc[im] * n_light
    return total_heavy / total_light if total_light > 0 else 0.0


def total_ratio_vectorized(conc: np.ndarray, n_atoms: int) -> float:
    """Vectorized compound-specific isotope ratio.

    Args:
        conc: Isotopologue concentration vector.
        n_atoms: Number of labelable atom positions.

    Returns:
        Bulk isotope ratio (heavy / light).
    """
    nm = 1 << n_atoms
    indices = np.arange(nm)
    # Count bits using lookup — faster than bin().count() in a loop
    n_heavy = np.zeros(nm, dtype=np.int32)
    for bit in range(n_atoms):
        n_heavy += (indices >> bit) & 1
    n_light = n_atoms - n_heavy

    weighted_heavy = np.dot(conc, n_heavy)
    weighted_light = np.dot(conc, n_light)
    return weighted_heavy / weighted_light if weighted_light > 0 else 0.0


def site_ratio(conc: np.ndarray, position: int) -> float:
    """Site-specific isotope ratio at a particular atom position.

    Args:
        conc: Isotopologue concentration vector of length 2^n.
        position: 0-indexed atom position.

    Returns:
        Ratio of heavy / light at this specific position.
    """
    nm = len(conc)
    n = int(np.log2(nm))
    bit = n - 1 - position  # bit position (MSB = position 0)
    mask = 1 << bit
    indices = np.arange(nm)
    heavy_mask = (indices & mask) != 0
    total_heavy = conc[heavy_mask].sum()
    total_light = conc[~heavy_mask].sum()
    return total_heavy / total_light if total_light > 0 else 0.0


def delta(ratio: float, element: str = "C") -> float:
    """Convert an isotope ratio to delta notation (permil).

    delta = (R_sample / R_standard - 1) * 1000

    Args:
        ratio: The measured isotope ratio.
        element: Element code for selecting the reference standard.

    Returns:
        Delta value in permil (‰).
    """
    ref = REFERENCE_RATIOS.get(element, VPDB_RATIO)
    return (ratio / ref - 1.0) * 1000.0


def site_delta(conc: np.ndarray, position: int, element: str = "C") -> float:
    """Site-specific delta value at a particular atom position.

    Args:
        conc: Isotopologue concentration vector.
        position: 0-indexed atom position.
        element: Element code for reference standard.

    Returns:
        Delta value in permil (‰).
    """
    return delta(site_ratio(conc, position), element)


def mass_spectrum(conc: np.ndarray) -> np.ndarray:
    """Mass isotopologue distribution (MID) — a.k.a. GetProp in QIRN.

    Groups isotopologues by total number of heavy atoms and returns the
    fractional abundance at each mass shift: [M+0, M+1, M+2, ...].

    Args:
        conc: Isotopologue concentration vector of length 2^n.

    Returns:
        Array of length n+1 with fractional abundances.
    """
    nm = len(conc)
    n = int(np.log2(nm))
    total = conc.sum()
    if total == 0:
        return np.zeros(n + 1)

    spectrum = np.zeros(n + 1)
    for im in range(nm):
        n_heavy = bin(im).count("1")
        spectrum[n_heavy] += conc[im]
    return spectrum / total


def mass_spectrum_vectorized(conc: np.ndarray, n_atoms: int) -> np.ndarray:
    """Vectorized mass isotopologue distribution.

    Args:
        conc: Isotopologue concentration vector.
        n_atoms: Number of labelable atom positions.

    Returns:
        Array of length n_atoms+1 with fractional abundances.
    """
    nm = 1 << n_atoms
    indices = np.arange(nm)
    n_heavy = np.zeros(nm, dtype=np.int32)
    for bit in range(n_atoms):
        n_heavy += (indices >> bit) & 1

    total = conc.sum()
    if total == 0:
        return np.zeros(n_atoms + 1)

    spectrum = np.zeros(n_atoms + 1)
    np.add.at(spectrum, n_heavy, conc)
    return spectrum / total


def natural_abundance(n_atoms: int, element: str = "C") -> np.ndarray:
    """Generate isotopologue concentrations at natural abundance.

    For each isotopologue with k heavy atoms among n positions, the
    probability is C(n,k) * p^k * (1-p)^(n-k), but the binary indexing
    distributes this across specific position patterns.

    The concentration of isotopologue ``im`` is::

        prod(p if bit set else (1-p) for each bit)

    Args:
        n_atoms: Number of labelable positions.
        element: Element code to determine natural abundance fraction.

    Returns:
        Isotopologue vector of length 2^n normalised to sum to 1.
    """
    ref = REFERENCE_RATIOS.get(element, VPDB_RATIO)
    p_heavy = ref / (1.0 + ref)
    p_light = 1.0 - p_heavy

    nm = 1 << n_atoms
    conc = np.ones(nm)
    for bit in range(n_atoms):
        mask = 1 << bit
        for im in range(nm):
            if im & mask:
                conc[im] *= p_heavy
            else:
                conc[im] *= p_light
    return conc


def natural_abundance_vectorized(n_atoms: int, element: str = "C") -> np.ndarray:
    """Vectorized natural abundance distribution.

    Args:
        n_atoms: Number of labelable positions.
        element: Element code.

    Returns:
        Isotopologue vector normalised to sum to 1.
    """
    ref = REFERENCE_RATIOS.get(element, VPDB_RATIO)
    p_heavy = ref / (1.0 + ref)
    p_light = 1.0 - p_heavy

    nm = 1 << n_atoms
    indices = np.arange(nm)
    conc = np.ones(nm)
    for bit in range(n_atoms):
        is_heavy = (indices >> bit) & 1
        conc *= np.where(is_heavy, p_heavy, p_light)
    return conc


def intramolecular(
    n_atoms: int,
    site_deltas: np.ndarray,
    total_conc: float = 1.0,
    element: str = "C",
) -> np.ndarray:
    """Build isotopologue vector from site-specific delta values.

    Converts per-site delta values (‰) to site-specific abundances, then
    builds the full isotopologue vector via the product of per-site abundances.

    This reimplements QIRN's ``Intramolecular`` function.

    Args:
        n_atoms: Number of labelable positions.
        site_deltas: Array of delta values (‰) for each position.
        total_conc: Total concentration to scale by.
        element: Element code for reference standard.

    Returns:
        Isotopologue vector of length 2^n, summing to total_conc.
    """
    ref = REFERENCE_RATIOS.get(element, VPDB_RATIO)
    # Convert delta (‰) → ratio → fractional abundance
    site_ratios = (site_deltas / 1000.0 + 1.0) * ref
    site_p_heavy = site_ratios / (site_ratios + 1.0)
    site_p_light = 1.0 - site_p_heavy

    nm = 1 << n_atoms
    conc = np.full(nm, total_conc)
    for bit in range(n_atoms):
        pos = (n_atoms - 1) - bit  # MSB = position 0
        mask = 1 << bit
        indices = np.arange(nm)
        is_heavy = (indices & mask).astype(bool)
        conc *= np.where(is_heavy, site_p_heavy[pos], site_p_light[pos])
    return conc
