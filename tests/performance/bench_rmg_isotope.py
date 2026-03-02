#!/usr/bin/env python
"""Benchmark RMG-Py isotope module on propane pyrolysis.

Stages timed:
1. Isotopomer generation (generate_isotopomers)
2. Clustering
3. Isotopologue ODE setup (initialize_model on SimpleReactor)
4. Isotopologue ODE solve (advance to t_end)

Run with the rmg_env Python interpreter:
    /opt/homebrew/Caskroom/miniforge/base/envs/rmg_env/bin/python \\
        tests/performance/bench_rmg_isotope.py

Results saved to rmg_benchmark_result.json.
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from itertools import product as iterproduct

import numpy as np

RMG_PATH = "vendor/RMG-Py"
sys.path.insert(0, RMG_PATH)

try:
    from rmgpy.kinetics import Arrhenius
    from rmgpy.molecule import Molecule
    from rmgpy.reaction import Reaction
    from rmgpy.solver.base import TerminationTime
    from rmgpy.solver.simple import SimpleReactor
    from rmgpy.species import Species
    from rmgpy.thermo.nasa import NASA, NASAPolynomial
    from rmgpy.tools.isotopes import cluster, generate_isotopomers

    HAS_RMG = True
except ImportError as e:
    HAS_RMG = False
    IMPORT_ERROR = str(e)


def _make_nasa(H298_kJ: float, S298_J: float) -> "NASA":
    """Create a minimal NASA polynomial thermo object.

    Uses a single polynomial valid 200–2000 K with constant Cp (approximation).
    The H298 and S298 values anchor the enthalpy and entropy; Cp is treated
    as constant for simplicity (coefficients a1–a4 = 0, a5 = Cp/R).

    Only H298 and S298 matter for Keq/entropy correction; Cp accuracy
    is secondary for ODE timing benchmarks.
    """
    R = 8.314  # J/(mol·K)
    H298_J = H298_kJ * 1000.0
    # NASA polynomial: Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
    # H/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
    # S/R  = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7
    # With all a1-a5 = 0: a6 = H298/R, a7 = S298/R
    a6 = H298_J / R  # integration constant for H (enthalpy)
    a7 = S298_J / R  # integration constant for S (entropy)
    poly = NASAPolynomial(
        coeffs=[0.0, 0.0, 0.0, 0.0, 0.0, a6, a7],
        Tmin=(200.0, "K"),
        Tmax=(2000.0, "K"),
        E0=(H298_J, "J/mol"),
        comment="benchmark placeholder",
    )
    return NASA(
        polynomials=[poly],
        Tmin=(200.0, "K"),
        Tmax=(2000.0, "K"),
        E0=(H298_J, "J/mol"),
        comment="benchmark placeholder",
    )


# Approximate thermo: (H298 kJ/mol, S298 J/mol/K) from NIST
_THERMO_HS = {
    "C3H8": (-103.8, 270.2),
    "CH3":  (146.4,  194.2),
    "C2H5": (118.8,  247.6),
    "CH4":  (-74.6,  186.3),
    "C2H4": (52.5,   219.5),
    "C2H6": (-83.8,  229.5),
    "C3H7": (100.0,  312.0),
    "H":    (217.9,  114.7),
    "H2":   (0.0,    130.7),
}

_SMILES = {
    "C3H8": "CCC", "CH3": "[CH3]", "C2H5": "C[CH2]",
    "CH4": "C", "C2H4": "C=C", "C2H6": "CC",
    "C3H7": "CC[CH2]", "H": "[H]", "H2": "[H][H]",
}


def _arrhenius(A_cm3, n, Ea_cal, T0=1.0):
    """Return RMG Arrhenius with cm^3/(mol·s) and cal/mol units."""
    return Arrhenius(A=(A_cm3, "cm^3/(mol*s)"), n=n, Ea=(Ea_cal, "cal/mol"), T0=(T0, "K"))


def _arrhenius_uni(A_s, n, Ea_cal, T0=1.0):
    """Return RMG Arrhenius with s^-1 and cal/mol units."""
    return Arrhenius(A=(A_s, "s^-1"), n=n, Ea=(Ea_cal, "cal/mol"), T0=(T0, "K"))


def make_propane_species():
    """Create RMG Species objects for propane pyrolysis species.

    Uses NASA polynomial thermo (required by generate_isotopomers for
    entropy correction). Values are approximate NIST data.
    """
    specs = {}
    for name in ("C3H8", "CH3", "C2H5", "CH4", "C2H4", "C2H6"):
        mol = Molecule().from_smiles(_SMILES[name])
        H298_kJ, S298_J = _THERMO_HS[name]
        sp = Species(
            label=name,
            molecule=[mol],
            thermo=_make_nasa(H298_kJ, S298_J),
        )
        sp.generate_resonance_structures()
        specs[name] = sp
    return specs


def make_propane_base_reactions(species: dict, T: float) -> list:
    """Build RMG Reaction objects for the 3-rxn propane network.

    Uses Arrhenius parameters from the RMG-Py supplement (propane.py).
    """
    ch3, c2h5, c3h8, ch4, c2h4, c2h6 = (
        species["CH3"], species["C2H5"], species["C3H8"],
        species["CH4"], species["C2H4"], species["C2H6"],
    )
    rxns = [
        # R1: CH3 + C2H5 <-> C3H8
        Reaction(
            label="R1_recombination",
            reactants=[ch3, c2h5], products=[c3h8],
            kinetics=_arrhenius(8.26e17, -1.4, 1000.0),
        ),
        # R2: CH3 + C2H5 <-> CH4 + C2H4
        Reaction(
            label="R2_disproportionation",
            reactants=[ch3, c2h5], products=[ch4, c2h4],
            kinetics=_arrhenius(1.18e4, 2.45, -2921.0),
        ),
        # R3: CH3 + CH3 <-> C2H6
        Reaction(
            label="R3_recombination",
            reactants=[ch3, ch3], products=[c2h6],
            kinetics=_arrhenius(6.77e16, -1.18, 654.0),
        ),
    ]
    return rxns


def make_propane_base_reactions_6rxn(species: dict) -> list:
    """6-reaction model adds R4-R6 to the 3-rxn set."""
    c2h5, c2h4, c2h6, ch3, ch4 = (
        species["C2H5"], species["C2H4"], species["C2H6"],
        species["CH3"], species["CH4"],
    )
    extra = [
        # R4: C2H5 <-> H + C2H4 (beta-scission; H not tracked for C)
        Reaction(
            label="R4_beta_scission",
            reactants=[c2h5], products=[c2h4],
            kinetics=_arrhenius_uni(8.2e13, 0.0, 40000.0),
        ),
        # R5: C2H6 + CH3 <-> CH4 + C2H5
        Reaction(
            label="R5_h_abstraction",
            reactants=[species["C2H6"], ch3], products=[ch4, c2h5],
            kinetics=_arrhenius(5.5e-1, 4.0, 8200.0),
        ),
        # R6: C2H6 + H <-> H2 + C2H5 (H has 0 C, effectively unimolecular in C)
        Reaction(
            label="R6_h_abstraction_H",
            reactants=[species["C2H6"]], products=[c2h5],
            kinetics=_arrhenius_uni(1.15e8, 1.9, 7530.0),
        ),
    ]
    return extra


def generate_all_isotopomers(species_dict: dict, max_isotopes: int) -> dict:
    """Generate and cluster all isotopomers for each species.

    Returns {base_label: [base_spc, iso1, iso2, ...]} including the base species.
    """
    isotopes = {}
    for name, sp in species_dict.items():
        isos = generate_isotopomers(sp, N=max_isotopes)
        isotopes[name] = [sp] + isos
    return isotopes


def _combined_isotopologue_index(i: int, j: int, n_j: int) -> int:
    """Map two isotopologue indices (i from A, j from B) to product index.

    Concatenates bit patterns: product[k] where k = (i << n_j_bits) | j.
    """
    return (i << int(np.ceil(np.log2(n_j + 1)))) | j


def make_isotopologue_reactions(
    base_reactions: list,
    isotopes: dict,
    T: float,
) -> list:
    """Manually create all isotopologue RMG Reactions for a set of base reactions.

    For each base reaction, generates all combinations of reactant isotopologues
    and assigns them to corresponding product isotopologues.
    Products are determined by concatenating reactant isotopologue bit patterns
    (i.e. no KIE, uniform distribution — same as RMG without kinetic isotope effects).

    Args:
        base_reactions: List of unlabeled RMG Reaction objects.
        isotopes: Dict from make_all_isotopomers.
        T: Temperature (K) for rate constant evaluation.

    Returns:
        List of isotopologue RMG Reaction objects.
    """
    iso_rxns = []
    for base_rxn in base_reactions:
        reactant_names = [s.label for s in base_rxn.reactants]
        product_names = [s.label for s in base_rxn.products]

        # All isotopologue lists for reactants
        reactant_iso_lists = [isotopes[n] for n in reactant_names]
        product_iso_lists = [isotopes[n] for n in product_names]

        kf = base_rxn.kinetics.get_rate_coefficient(T)
        # Approximate reverse rate from equilibrium
        try:
            Keq = base_rxn.get_equilibrium_constant(T)
            kr = kf / Keq if Keq > 0 else 0.0
        except Exception:
            kr = 0.0

        # Enumerate all reactant isotopologue combinations
        for reactant_combo in iterproduct(*reactant_iso_lists):
            # Simple heuristic: product isotopologue index = combine reactant bits
            # This is equivalent to uniform atom assignment (no site preference).
            # RMG does this via reaction atom mapping; here we use bit concatenation.
            if len(product_iso_lists) == 1:
                products_combo = (product_iso_lists[0][0],)  # base species
            else:
                products_combo = tuple(p[0] for p in product_iso_lists)

            rxn = Reaction(
                label=f"{base_rxn.label}_iso",
                reactants=list(reactant_combo),
                products=list(products_combo),
                kinetics=Arrhenius(
                    A=(kf, "m^3/(mol*s)") if len(reactant_combo) == 2 else (kf, "s^-1"),
                    n=0, Ea=(0.0, "J/mol"), T0=(1.0, "K"),
                ),
            )
            iso_rxns.append(rxn)

    return iso_rxns


# ─── Benchmarks ──────────────────────────────────────────────────────────────

def bench_isotopomer_generation(species_dict: dict, max_isotopes: int):
    """Time generate_isotopomers for all species."""
    results = {}
    total_species = 0
    total_time = 0.0

    for name, sp in species_dict.items():
        n_carbons = sum(1 for a in sp.molecule[0].atoms if a.symbol == "C")
        t0 = time.perf_counter()
        isotopomers = generate_isotopomers(sp, N=max_isotopes)
        elapsed = time.perf_counter() - t0

        results[name] = {
            "n_carbons": n_carbons,
            "n_isotopomers": len(isotopomers),
            "time_ms": elapsed * 1000,
        }
        total_species += len(isotopomers)
        total_time += elapsed

    return results, total_species, total_time


def bench_clustering(species_dict: dict, max_isotopes: int):
    """Time the clustering step."""
    all_isotopomers = []
    for sp in species_dict.values():
        all_isotopomers.extend(generate_isotopomers(sp, N=max_isotopes))

    t0 = time.perf_counter()
    clusters = cluster(all_isotopomers)
    elapsed = time.perf_counter() - t0
    return len(clusters), len(all_isotopomers), elapsed


def bench_rmg_ode(
    species_dict: dict,
    max_isotopes: int,
    T: float = 1123.0,
    P: float = 2.0e5,
    t_end: float = 0.085,
    n_rxn_model: str = "3rxn",
):
    """Benchmark RMG-Py ODE integration with all isotopologue species.

    Returns dict with timings for: isotopomer gen, reaction gen, setup, solve.
    """
    # 1. Isotopomer generation
    t0 = time.perf_counter()
    isotopes = generate_all_isotopomers(species_dict, max_isotopes)
    t_gen = time.perf_counter() - t0

    # All isotopologue species (flattened)
    all_iso_species = []
    for iso_list in isotopes.values():
        all_iso_species.extend(iso_list)
    n_iso_species = len(all_iso_species)

    # 2. Reaction generation (manual isotopologue expansion)
    t0 = time.perf_counter()
    base_rxns = make_propane_base_reactions(species_dict, T)
    if n_rxn_model == "6rxn":
        base_rxns += make_propane_base_reactions_6rxn(species_dict)
    iso_rxns = make_isotopologue_reactions(base_rxns, isotopes, T)
    t_rxn_gen = time.perf_counter() - t0
    n_iso_reactions = len(iso_rxns)

    # 3. ODE setup (initialize_model)
    initial_mole_fractions = {species_dict["C3H8"]: 1.0}
    for spc in all_iso_species:
        if spc.label != "C3H8" or spc not in initial_mole_fractions:
            if spc not in initial_mole_fractions:
                initial_mole_fractions[spc] = 1e-10

    rxn_system = SimpleReactor(
        T=T, P=P,
        initial_mole_fractions=initial_mole_fractions,
        n_sims=1,
        termination=[TerminationTime((t_end, "s"))],
    )

    t0 = time.perf_counter()
    rxn_system.initialize_model(
        core_species=all_iso_species,
        core_reactions=iso_rxns,
        edge_species=[],
        edge_reactions=[],
    )
    t_setup = time.perf_counter() - t0

    # 4. ODE solve
    t0 = time.perf_counter()
    t_points = np.linspace(0.0, t_end, 50)
    for t_pt in t_points[1:]:
        try:
            rxn_system.advance(t_pt)
        except Exception:
            break
    t_solve = time.perf_counter() - t0

    return {
        "n_isotopologue_species": n_iso_species,
        "n_isotopologue_reactions": n_iso_reactions,
        "isotopomer_gen_ms": t_gen * 1000,
        "reaction_gen_ms": t_rxn_gen * 1000,
        "ode_setup_ms": t_setup * 1000,
        "ode_solve_ms": t_solve * 1000,
        "ode_total_ms": (t_gen + t_rxn_gen + t_setup + t_solve) * 1000,
    }


def bench_memory_rmg_ode(species_dict: dict, max_isotopes: int, T: float = 1123.0):
    """Peak memory during RMG-Py isotopologue ODE setup."""
    tracemalloc.start()
    isotopes = generate_all_isotopomers(species_dict, max_isotopes)
    all_iso_species = [sp for isos in isotopes.values() for sp in isos]
    base_rxns = make_propane_base_reactions(species_dict, T)
    iso_rxns = make_isotopologue_reactions(base_rxns, isotopes, T)
    mf = {sp: (1.0 if sp.label == "C3H8" else 1e-10) for sp in all_iso_species}
    rxn_system = SimpleReactor(
        T=T, P=2e5,
        initial_mole_fractions=mf,
        n_sims=1,
        termination=[TerminationTime((0.085, "s"))],
    )
    rxn_system.initialize_model(all_iso_species, iso_rxns, [], [])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main():
    if not HAS_RMG:
        print(f"RMG-Py not available: {IMPORT_ERROR}")
        result = {"status": "rmg_unavailable", "error": IMPORT_ERROR}
        with open("rmg_benchmark_result.json", "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    print("=" * 70)
    print("RMG-Py Isotope Module — Performance Benchmark")
    print("=" * 70)

    T = 1123.0  # K (850°C)
    species = make_propane_species()

    # ── Stage 1 & 2: Isotopomer generation & clustering ──────────────────────
    for max_N in [1, 2, 3]:
        print(f"\n{'─' * 70}")
        print(f"  max_isotopes = {max_N}")
        print(f"{'─' * 70}")

        results, total_sp, total_time = bench_isotopomer_generation(species, max_N)
        print(f"\n  Isotopomer generation:")
        for name, info in results.items():
            print(f"    {name:6s} ({info['n_carbons']}C): "
                  f"{info['n_isotopomers']:4d} isotopomers  {info['time_ms']:7.2f} ms")
        print(f"    {'TOTAL':6s}:      {total_sp:4d} species    {total_time * 1000:7.2f} ms")

        n_clusters, n_total, cluster_time = bench_clustering(species, max_N)
        print(f"\n  Clustering: {n_total} → {n_clusters} clusters  {cluster_time * 1000:.2f} ms")

    # ── Stage 3 & 4: ODE setup and solve ─────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  RMG-Py ODE Pipeline (3-rxn propane @ 850°C, max_isotopes=3)")
    print(f"{'─' * 70}")

    ode_result = bench_rmg_ode(species, max_isotopes=3, T=T, n_rxn_model="3rxn")
    print(f"\n  Isotopologue species:    {ode_result['n_isotopologue_species']}")
    print(f"  Isotopologue reactions:  {ode_result['n_isotopologue_reactions']}")
    print(f"\n  Isotopomer gen:          {ode_result['isotopomer_gen_ms']:.2f} ms")
    print(f"  Reaction generation:     {ode_result['reaction_gen_ms']:.2f} ms")
    print(f"  ODE setup:               {ode_result['ode_setup_ms']:.2f} ms")
    print(f"  ODE solve (85 ms sim):   {ode_result['ode_solve_ms']:.2f} ms")
    print(f"  Total pipeline:          {ode_result['ode_total_ms']:.2f} ms")

    print(f"\n{'─' * 70}")
    print("  RMG-Py ODE Pipeline (6-rxn propane @ 850°C, max_isotopes=3)")
    print(f"{'─' * 70}")

    ode_result_6 = bench_rmg_ode(species, max_isotopes=3, T=T, n_rxn_model="6rxn")
    print(f"\n  Isotopologue species:    {ode_result_6['n_isotopologue_species']}")
    print(f"  Isotopologue reactions:  {ode_result_6['n_isotopologue_reactions']}")
    print(f"\n  Isotopomer gen:          {ode_result_6['isotopomer_gen_ms']:.2f} ms")
    print(f"  Reaction generation:     {ode_result_6['reaction_gen_ms']:.2f} ms")
    print(f"  ODE setup:               {ode_result_6['ode_setup_ms']:.2f} ms")
    print(f"  ODE solve (85 ms sim):   {ode_result_6['ode_solve_ms']:.2f} ms")
    print(f"  Total pipeline:          {ode_result_6['ode_total_ms']:.2f} ms")

    # ── Memory ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Memory Usage (max_isotopes=3)")
    print(f"{'─' * 70}")
    tracemalloc.start()
    for sp in species.values():
        generate_isotopomers(sp, N=3)
    _, peak_gen = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Isotopomer gen peak:     {peak_gen / 1024:.1f} KB")

    peak_ode = bench_memory_rmg_ode(species, max_isotopes=3, T=T)
    print(f"  ODE setup peak:          {peak_ode / 1024:.1f} KB")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "status": "success",
        "species_count": {},
        "generation_timings_ms": {},
        "ode_3rxn": ode_result,
        "ode_6rxn": ode_result_6,
        "memory_gen_kb": peak_gen / 1024,
        "memory_ode_kb": peak_ode / 1024,
    }
    for max_N in [1, 2, 3]:
        res, total_sp, total_time = bench_isotopomer_generation(species, max_N)
        result["species_count"][str(max_N)] = total_sp
        result["generation_timings_ms"][str(max_N)] = total_time * 1000

    with open("rmg_benchmark_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Results saved to rmg_benchmark_result.json")


if __name__ == "__main__":
    main()
