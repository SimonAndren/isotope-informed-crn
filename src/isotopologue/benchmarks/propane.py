"""Propane pyrolysis benchmark networks from the RMG-Py isotope supplement.

Defines 3-reaction, 6-reaction, and 18-reaction DRG models for propane
pyrolysis at high temperature. Rate constants are Arrhenius parameters
from the supplement Tables S1-S6.

Conditions: T = 750-950 C (1023-1223 K), P = 2 bar, ~85 ms residence time.
"""

from __future__ import annotations

import numpy as np

from isotopologue.analysis import intramolecular, natural_abundance_vectorized
from isotopologue.species import Network, Reaction, Species


def arrhenius(A: float, n: float, Ea: float, T: float) -> float:
    """Arrhenius rate constant: k = A * T^n * exp(-Ea / RT).

    Args:
        A: Pre-exponential factor.
        n: Temperature exponent.
        Ea: Activation energy in cal/mol.
        T: Temperature in K.

    Returns:
        Rate constant in appropriate units.
    """
    R = 1.987  # cal/(mol·K)
    return A * T**n * np.exp(-Ea / (R * T))


# ─── Species definitions ────────────────────────────────────────────────────
# Only carbon positions are tracked for C-13. H2, H, He have 0 labeled atoms.

PROPANE_SPECIES = {
    "C3H8": Species("C3H8", n_labeled=3, element="C"),
    "CH3": Species("CH3", n_labeled=1, element="C"),
    "C2H5": Species("C2H5", n_labeled=2, element="C"),
    "CH4": Species("CH4", n_labeled=1, element="C"),
    "C2H4": Species("C2H4", n_labeled=2, element="C"),
    "C2H6": Species("C2H6", n_labeled=2, element="C"),
    "H": Species("H", n_labeled=0, element="C"),
    "H2": Species("H2", n_labeled=0, element="C"),
    "C3H7": Species("C3H7", n_labeled=3, element="C"),
}


def _make_reactions_3rxn(T: float) -> list[Reaction]:
    """3-reaction propane pyrolysis model (R1-R3).

    R1: CH3 + C2H5 <=> C3H8    (recombination)
    R2: CH3 + C2H5 <=> CH4 + C2H4  (disproportionation)
    R3: CH3 + CH3 <=> C2H6     (recombination)

    Arrhenius parameters from RMG-Py/NIST for propane pyrolysis.
    """
    # R1: CH3 + C2H5 → C3H8 (recombination)
    # A in cm^3/(mol·s), typical recombination
    k1f = arrhenius(8.26e17, -1.4, 1000.0, T)
    k1r = arrhenius(7.92e22, -1.7, 87700.0, T)
    n_combo = 1 + 2  # CH3(1C) + C2H5(2C) = 3C combined
    R1 = Reaction(
        "R1_recombination",
        reactants=("CH3", "C2H5"),
        products=("C3H8",),
        k_forward=np.full(1 << n_combo, k1f),
        k_reverse=np.full(1 << n_combo, k1r),
    )

    # R2: CH3 + C2H5 → CH4 + C2H4 (disproportionation)
    k2f = arrhenius(1.18e4, 2.45, -2921.0, T)
    k2r = arrhenius(6.02e11, 0.0, 65000.0, T)
    n_react = 1 + 2  # 3C combined reactant space
    R2 = Reaction(
        "R2_disproportionation",
        reactants=("CH3", "C2H5"),
        products=("CH4", "C2H4"),
        k_forward=np.full(1 << n_react, k2f),
        k_reverse=np.full(1 << n_react, k2r),
    )

    # R3: CH3 + CH3 → C2H6 (recombination)
    k3f = arrhenius(6.77e16, -1.18, 654.0, T)
    k3r = arrhenius(4.73e19, -1.2, 90600.0, T)
    n_combo3 = 1 + 1  # 2C combined
    R3 = Reaction(
        "R3_recombination",
        reactants=("CH3", "CH3"),
        products=("C2H6",),
        k_forward=np.full(1 << n_combo3, k3f),
        k_reverse=np.full(1 << n_combo3, k3r),
    )

    return [R1, R2, R3]


def _make_reactions_6rxn(T: float) -> list[Reaction]:
    """6-reaction model adds R4-R6 to R1-R3.

    R4: C2H5 <=> H + C2H4
    R5: C2H6 + CH3 <=> CH4 + C2H5
    R6: C2H6 + H <=> H2 + C2H5
    """
    rxns = _make_reactions_3rxn(T)

    # R4: C2H5 → H + C2H4 (beta-scission)
    k4f = arrhenius(8.2e13, 0.0, 40000.0, T)
    k4r = arrhenius(1.08e12, 0.0, 1800.0, T)
    # Simple reaction for C: C2H5(2C) → C2H4(2C), H has 0C
    # H is not tracked for C-13, so this is effectively A(2C) → C(2C)
    R4 = Reaction(
        "R4_beta_scission",
        reactants=("C2H5",),
        products=("C2H4",),
        k_forward=np.full(4, k4f),  # 2^2
        k_reverse=np.full(4, k4r),
    )

    # R5: C2H6 + CH3 → CH4 + C2H5 (H-abstraction)
    k5f = arrhenius(5.5e-1, 4.0, 8200.0, T)
    k5r = arrhenius(5.0e-1, 4.0, 12200.0, T)
    n_react = 2 + 1  # 3C combined
    R5 = Reaction(
        "R5_h_abstraction",
        reactants=("C2H6", "CH3"),
        products=("CH4", "C2H5"),
        k_forward=np.full(1 << n_react, k5f),
        k_reverse=np.full(1 << n_react, k5r),
    )

    # R6: C2H6 + H → H2 + C2H5 (H-abstraction by H radical)
    k6f = arrhenius(1.15e8, 1.9, 7530.0, T)
    k6r = arrhenius(3.62e6, 2.0, 11100.0, T)
    # H has 0 labeled carbons, so the "synthesis" is just C2H6(2C) → C2H5(2C)
    R6 = Reaction(
        "R6_h_abstraction_H",
        reactants=("C2H6",),
        products=("C2H5",),
        k_forward=np.full(4, k6f),
        k_reverse=np.full(4, k6r),
    )

    rxns.extend([R4, R5, R6])
    return rxns


def propane_3rxn(T: float = 1123.0) -> Network:
    """Build the 3-reaction propane pyrolysis network.

    Args:
        T: Temperature in K (default 1123K = 850°C).

    Returns:
        Network ready for integration.
    """
    species = {
        name: sp
        for name, sp in PROPANE_SPECIES.items()
        if name in ("C3H8", "CH3", "C2H5", "CH4", "C2H4", "C2H6")
    }
    return Network(species=species, reactions=_make_reactions_3rxn(T))


def propane_6rxn(T: float = 1123.0) -> Network:
    """Build the 6-reaction propane pyrolysis network.

    Args:
        T: Temperature in K.

    Returns:
        Network ready for integration.
    """
    species = {
        name: sp
        for name, sp in PROPANE_SPECIES.items()
        if name in ("C3H8", "CH3", "C2H5", "CH4", "C2H4", "C2H6")
    }
    return Network(species=species, reactions=_make_reactions_6rxn(T))


def initial_conditions(
    network: Network,
    propane_conc: float = 1.0,
    site_preference: float = 5.45,
) -> dict[str, np.ndarray]:
    """Generate initial isotopologue concentrations for propane pyrolysis.

    Args:
        network: The reaction network.
        propane_conc: Total propane concentration.
        site_preference: Propane site-specific delta difference (‰).
            Distributed as: C1=+sp/3, C2=-sp*2/3, C3=+sp/3.

    Returns:
        Dictionary of species name → isotopologue concentration vector.
    """
    conc = {}
    for name, sp in network.species.items():
        if name == "C3H8":
            # Propane: apply site preference
            d1 = site_preference / 3.0
            d2 = -site_preference * 2.0 / 3.0
            d3 = site_preference / 3.0
            conc[name] = intramolecular(3, np.array([d1, d2, d3]), propane_conc)
        elif sp.n_labeled > 0:
            # Other species: trace amounts at natural abundance
            conc[name] = natural_abundance_vectorized(sp.n_labeled) * 1e-10
        else:
            # Non-carbon species (H, H2): single "isotopologue"
            conc[name] = np.array([1e-10])
    return conc
