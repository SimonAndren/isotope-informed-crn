"""Kinetic isotope effect (KIE) rate vector construction.

Builds rate constant vectors of length 2^n where each entry accounts for
the site-specific kinetic isotope effects at the labeled positions.
"""

from __future__ import annotations

import numpy as np


def build_rate_vector(
    n_atoms: int,
    base_rate: float,
    site_kies: dict[int, float] | None = None,
) -> np.ndarray:
    """Build a rate constant vector with per-site KIE.

    For a molecule with n_atoms labelable positions, produces a vector of
    length 2^n_atoms. Each entry is the effective rate constant for an
    isotopologue whose heavy-atom pattern is given by the binary representation
    of the index.

    The monoisotopic (all-light) rate is ``base_rate``. For multiply-substituted
    isotopologues, the rate is::

        k[i] = base_rate * prod(kie[j] for j in heavy_positions(i)) / base_rate^(n_heavy - 1)

    This matches QIRN's GetFullRates normalisation convention.

    Args:
        n_atoms: Number of labelable atom positions.
        base_rate: Rate for the monoisotopic (all-light) species.
        site_kies: Mapping from atom position (0-indexed) to KIE factor.
            Positions not listed default to 1.0 (no fractionation).

    Returns:
        float64 array of length 2^n_atoms.
    """
    nm = 1 << n_atoms
    if site_kies is None or len(site_kies) == 0:
        return np.full(nm, base_rate)

    rates = np.empty(nm)
    rates[0] = base_rate

    for im in range(1, nm):
        product = 1.0
        n_heavy = 0
        for pos in range(n_atoms):
            bit = n_atoms - 1 - pos  # high bit = position 0
            if (im >> bit) & 1:
                n_heavy += 1
                product *= site_kies.get(pos, 1.0)
        # Normalise: single-sub rates are base_rate * kie[pos],
        # multi-sub rates divide out extra base_rate factors.
        rates[im] = (
            base_rate * product / (base_rate ** max(n_heavy - 1, 0)) if base_rate != 0 else 0.0
        )

    return rates


def reduced_mass_kie(
    base_mass: float,
    isotope_mass: float,
    hydrogen_mass: float = 1.00794,
    three_member: bool = True,
) -> float:
    """Compute the simple KIE factor from reduced mass ratio.

    Implements the Melander & Saunders (1980) approach used by RMG-py.

    For a three-member transition state (A-H-B), the reduced mass involves
    the hydrogen mass and the combined mass of the heavy atoms:
        mu = 1/m_H + 1/m_combined

    For a two-member transition state:
        mu = 1/m_A + 1/m_B

    The KIE is sqrt(mu_light / mu_heavy).

    Args:
        base_mass: Mass of the light isotope (e.g. 12 for C-12).
        isotope_mass: Mass of the heavy isotope (e.g. 13 for C-13).
        hydrogen_mass: Mass of hydrogen (default from RMG).
        three_member: Whether to use 3-member TS approximation.

    Returns:
        KIE factor (always >= 1 for normal KIE with heavier isotope).
    """
    if three_member:
        mu_light = 1.0 / (1.0 / hydrogen_mass + 1.0 / base_mass)
        mu_heavy = 1.0 / (1.0 / hydrogen_mass + 1.0 / isotope_mass)
    else:
        mu_light = base_mass
        mu_heavy = isotope_mass
    return (mu_light / mu_heavy) ** 0.5
