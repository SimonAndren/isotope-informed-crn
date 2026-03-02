"""Bridge: convert RMG-Py base reaction network to isotopologue.species.Network.

Takes unlabeled (base) RMG-Py Species and Reaction objects and constructs the
vector isotopologue Network ready for IsotopologueEngine.

This enables the workflow:
    1. Generate a reaction network with RMG-Py (chemistry discovery, thermo, KIEs)
    2. Convert to the vector format with rmg_network_to_vector()
    3. Solve isotopologue ODEs with IsotopologueEngine (orders-of-magnitude faster)

Usage:
    import sys; sys.path.insert(0, 'vendor/RMG-Py')
    from rmgpy.species import Species
    from rmgpy.reaction import Reaction
    from isotopologue.rmg_bridge import rmg_network_to_vector
    from isotopologue.engine import IsotopologueEngine

    net = rmg_network_to_vector(core_species, core_reactions, T=1123.0)
    engine = IsotopologueEngine(net)
"""

from __future__ import annotations

import numpy as np

from isotopologue.species import AtomMap, Network, Reaction, Species, build_atom_map

try:
    from rmgpy.species import Species as RMGSpecies
    from rmgpy.reaction import Reaction as RMGReaction

    HAS_RMG = True
except ImportError:
    HAS_RMG = False


def rmg_species_to_vector(rmg_spc: "RMGSpecies", element: str = "C") -> Species:
    """Convert an unlabeled RMG Species to a vector Species.

    Counts element atoms in the first molecule structure to determine n_labeled.

    Args:
        rmg_spc: Unlabeled RMG Species object (no isotopic substitution).
        element: Which element to track (default 'C' for carbon-13).

    Returns:
        isotopologue.species.Species with n_labeled = number of element atoms.
    """
    mol = rmg_spc.molecule[0]
    n_labeled = sum(1 for a in mol.atoms if a.symbol == element)
    return Species(name=rmg_spc.label, n_labeled=n_labeled, element=element)


def _reaction_type(n_reactants: int, n_products: int) -> str:
    if n_reactants == 1 and n_products == 1:
        return "simple"
    if n_reactants == 1 and n_products == 2:
        return "breakdown"
    if n_reactants == 2 and n_products == 1:
        return "synthesis"
    if n_reactants == 2 and n_products == 2:
        return "exchange"
    raise ValueError(f"Unsupported stoichiometry: {n_reactants}R → {n_products}P")


def _k_forward(rmg_rxn: "RMGReaction", T: float) -> float:
    """Extract forward rate constant at temperature T (SI units: m^3/mol/s or s^-1)."""
    return rmg_rxn.kinetics.get_rate_coefficient(T)


def _k_reverse(rmg_rxn: "RMGReaction", T: float) -> float:
    """Extract reverse rate constant via equilibrium constant.

    Falls back to 0.0 if thermo is unavailable (prevents division by zero).
    """
    kf = _k_forward(rmg_rxn, T)
    try:
        Keq = rmg_rxn.get_equilibrium_constant(T)
        return kf / Keq if Keq > 0 else 0.0
    except Exception:
        return 0.0


def _n_combined(labels: tuple[str, ...], species_map: dict[str, Species]) -> int:
    """Product of n_isotopologues across a list of species labels."""
    result = 1
    for label in labels:
        result *= species_map[label].n_isotopologues
    return result


def build_atom_map_from_rmg(
    rmg_rxn: "RMGReaction",
    reactant_labels: tuple[str, ...],
    product_labels: tuple[str, ...],
    species_map: dict[str, Species],
    element: str = "C",
) -> AtomMap | None:
    """Build an AtomMap from RMG's atom-pair mapping information.

    RMG stores atom-level mappings in rxn.pairs (list of (reactant_atom,
    product_atom) tuples). This function extracts the C-atom permutation
    from reactant isotopologue space to product isotopologue space.

    Returns None if pairs are unavailable or mapping cannot be determined.
    The engine handles None atom_map by using identity ordering.

    Args:
        rmg_rxn: RMG Reaction with populated pairs attribute.
        reactant_labels: Ordered tuple of reactant species labels.
        product_labels: Ordered tuple of product species labels.
        species_map: Dict label → vector Species.
        element: Element to track.

    Returns:
        AtomMap with forward and reverse permutation arrays, or None.
    """
    if not hasattr(rmg_rxn, "pairs") or not rmg_rxn.pairs:
        return None

    # Collect all C-atoms in reactant order (first species, then second)
    reactant_c_atoms = []
    for label in reactant_labels:
        rmg_sp = next(
            (s for s in rmg_rxn.reactants if s.label == label), None
        )
        if rmg_sp is None:
            return None
        for atom in rmg_sp.molecule[0].atoms:
            if atom.symbol == element:
                reactant_c_atoms.append(atom)

    product_c_atoms = []
    for label in product_labels:
        rmg_sp = next(
            (s for s in rmg_rxn.products if s.label == label), None
        )
        if rmg_sp is None:
            return None
        for atom in rmg_sp.molecule[0].atoms:
            if atom.symbol == element:
                product_c_atoms.append(atom)

    if len(reactant_c_atoms) != len(product_c_atoms):
        return None

    n = len(reactant_c_atoms)
    if n == 0:
        return None

    # Build forward mapping: for each product C position p, find the reactant position r
    # such that pair (reactant_c_atoms[r], product_c_atoms[p]) exists.
    pair_map: dict[int, int] = {}  # reactant_idx → product_idx
    for r_atom, p_atom in rmg_rxn.pairs:
        if r_atom.symbol == element and p_atom.symbol == element:
            try:
                r_idx = reactant_c_atoms.index(r_atom)
                p_idx = product_c_atoms.index(p_atom)
                pair_map[r_idx] = p_idx
            except ValueError:
                continue

    if len(pair_map) != n:
        return None

    # Convert to bit-position lists for build_atom_map
    # Bit position for atom at index i: (n-1-i) so atom 0 is MSB
    old_bits = [n - 1 - r for r in range(n)]
    new_bits = [n - 1 - pair_map[r] for r in range(n)]

    try:
        forward = build_atom_map(old_bits, new_bits)
        reverse = build_atom_map(new_bits, old_bits)
        return AtomMap(forward=forward, reverse=reverse)
    except Exception:
        return None


def rmg_reaction_to_vector(
    rmg_rxn: "RMGReaction",
    species_map: dict[str, Species],
    T: float,
    element: str = "C",
    rxn_label: str | None = None,
) -> Reaction:
    """Convert an unlabeled RMG Reaction to a vector Reaction.

    Builds uniform rate vectors (same rate for all isotopologues — no KIE).
    Atom map is extracted from rmg_rxn.pairs if available, else None.

    Args:
        rmg_rxn: Unlabeled RMG Reaction (not an isotopologue reaction).
        species_map: Dict label → vector Species (from rmg_species_to_vector).
        T: Temperature in K.
        element: Element to track.
        rxn_label: Override for reaction name (uses rmg_rxn.label if None).

    Returns:
        isotopologue.species.Reaction with k_forward and k_reverse vectors.
    """
    reactant_labels = tuple(s.label for s in rmg_rxn.reactants)
    product_labels = tuple(s.label for s in rmg_rxn.products)

    # Combined isotopologue space sizes
    n_fwd = _n_combined(reactant_labels, species_map)
    n_rev = _n_combined(product_labels, species_map)

    kf = _k_forward(rmg_rxn, T)
    kr = _k_reverse(rmg_rxn, T)

    name = rxn_label or getattr(rmg_rxn, "label", None) or f"rxn_{id(rmg_rxn)}"

    atom_map = build_atom_map_from_rmg(
        rmg_rxn, reactant_labels, product_labels, species_map, element
    )

    return Reaction(
        name=name,
        reactants=reactant_labels,
        products=product_labels,
        k_forward=np.full(n_fwd, kf),
        k_reverse=np.full(n_rev, kr),
        atom_map=atom_map,
    )


def rmg_network_to_vector(
    core_species: list["RMGSpecies"],
    core_reactions: list["RMGReaction"],
    T: float,
    element: str = "C",
) -> Network:
    """Convert an RMG-Py base reaction network to an isotopologue vector Network.

    Takes the UNLABELED core species and reactions from RMG-Py and constructs
    the full vector Network. The vector engine then propagates all 2^n
    isotopologue concentrations simultaneously without enumerating them explicitly.

    Args:
        core_species: List of unlabeled RMG Species (no ¹³C enrichment).
        core_reactions: List of unlabeled RMG Reactions (kinetics + thermo).
        T: Temperature in K for rate constant evaluation.
        element: Element to track (default 'C').

    Returns:
        isotopologue.species.Network ready for IsotopologueEngine.
    """
    # Build species map
    species_dict: dict[str, Species] = {}
    for rmg_spc in core_species:
        sp = rmg_species_to_vector(rmg_spc, element)
        species_dict[sp.name] = sp

    # Build reactions
    reactions: list[Reaction] = []
    for i, rmg_rxn in enumerate(core_reactions):
        label = getattr(rmg_rxn, "label", None) or f"R{i}"
        rxn = rmg_reaction_to_vector(rmg_rxn, species_dict, T, element, rxn_label=label)
        reactions.append(rxn)

    return Network(species=species_dict, reactions=reactions)
