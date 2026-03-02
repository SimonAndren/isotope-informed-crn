"""Performance benchmark for the QIRN vector isotopologue engine.

Measures wall time, RHS call count, and memory for propane pyrolysis networks
at different sizes and conditions. Also estimates what the equivalent RMG-py
isotope approach would require in terms of object counts and operations.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from isotopologue.benchmarks.propane import (
    initial_conditions,
    propane_3rxn,
    propane_6rxn,
)
from isotopologue.engine import IsotopologueEngine


def bench_rhs_call(engine: IsotopologueEngine, y0: np.ndarray, n_calls: int = 10_000):
    """Time a single RHS evaluation."""
    # Warm up
    for _ in range(100):
        engine.rhs(0.0, y0)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        engine.rhs(0.0, y0)
    elapsed = time.perf_counter() - t0
    return elapsed / n_calls


def bench_solve(engine: IsotopologueEngine, y0: np.ndarray, t_end: float, **kwargs):
    """Time a full ODE solve and return (elapsed, n_rhs_evals, result)."""
    # Count RHS calls
    call_count = [0]
    orig_rhs = engine.rhs

    def counting_rhs(t, y):
        call_count[0] += 1
        return orig_rhs(t, y)

    engine_rhs_backup = engine.rhs
    engine.rhs = counting_rhs

    t0 = time.perf_counter()
    result = engine.solve(y0, (0, t_end), **kwargs)
    elapsed = time.perf_counter() - t0

    engine.rhs = engine_rhs_backup
    return elapsed, call_count[0], result


def bench_memory(network_factory, t_end=0.085):
    """Measure peak memory for full solve."""
    tracemalloc.start()
    net = network_factory()
    engine = IsotopologueEngine(network=net)
    conc = initial_conditions(net)
    y0 = net.pack(conc)
    engine.solve(y0, (0, t_end), method="BDF", rtol=1e-8, atol=1e-12)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def count_state_variables(net):
    """Count total state variables (isotopologue concentrations)."""
    return sum(sp.n_isotopologues for sp in net.species.values())


def rmg_equivalent_counts(net):
    """Estimate what RMG-py would produce for the same network.

    RMG-py creates one Species object per isotopologue, and for each
    original reaction, generates all isotopologue reaction combinations.
    """
    n_species = sum(sp.n_isotopologues for sp in net.species.values())

    n_reactions = 0
    for rxn in net.reactions:
        rt = rxn.reaction_type
        if rt == "simple":
            # Each isotopologue of A maps to one reaction
            n_reactions += net.species[rxn.reactants[0]].n_isotopologues
        elif rt == "synthesis" or rt == "exchange":
            # Every combination of reactant isotopologues
            n_a = net.species[rxn.reactants[0]].n_isotopologues
            n_b = net.species[rxn.reactants[1]].n_isotopologues
            n_reactions += n_a * n_b
        elif rt == "breakdown":
            n_reactions += net.species[rxn.reactants[0]].n_isotopologues

    return n_species, n_reactions


def main():
    print("=" * 70)
    print("QIRN Vector Engine — Performance Benchmark")
    print("=" * 70)

    configs = [
        ("3-rxn propane (850°C)", propane_3rxn, 1123.0),
        ("6-rxn propane (850°C)", propane_6rxn, 1123.0),
        ("3-rxn propane (950°C)", propane_3rxn, 1223.0),
        ("6-rxn propane (950°C)", propane_6rxn, 1223.0),
    ]

    for label, factory, T in configs:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        net = factory(T)
        engine = IsotopologueEngine(network=net)
        conc = initial_conditions(net)
        y0 = net.pack(conc)
        n_state = count_state_variables(net)
        rmg_sp, rmg_rxn = rmg_equivalent_counts(net)

        print(f"  Species:              {len(net.species)}")
        print(f"  Reactions:            {len(net.reactions)}")
        print(f"  State variables:      {n_state}")
        print(f"  State vector bytes:   {y0.nbytes}")
        print()
        print(f"  RMG-py equiv species: {rmg_sp}")
        print(f"  RMG-py equiv rxns:    {rmg_rxn}")
        print(f"  Compression ratio:    {rmg_sp / len(net.species):.1f}x species, "
              f"{rmg_rxn / len(net.reactions):.1f}x reactions")
        print()

        # RHS timing
        rhs_time = bench_rhs_call(engine, y0, n_calls=10_000)
        print(f"  Single RHS call:      {rhs_time * 1e6:.1f} µs")

        # Short integration (0.1 ms)
        elapsed_short, calls_short, res_short = bench_solve(
            engine, y0, 1e-4, method="BDF", rtol=1e-8, atol=1e-12
        )
        assert res_short.success
        print(f"  0.1 ms integration:   {elapsed_short * 1e3:.1f} ms "
              f"({calls_short} RHS evals)")

        # Full 85 ms integration
        elapsed_full, calls_full, res_full = bench_solve(
            engine, y0, 0.085, method="BDF", rtol=1e-8, atol=1e-12
        )
        assert res_full.success
        print(f"  85 ms integration:    {elapsed_full * 1e3:.1f} ms "
              f"({calls_full} RHS evals)")

        # Throughput
        throughput = calls_full / elapsed_full
        print(f"  RHS throughput:       {throughput:.0f} evals/s")

    # Memory benchmark
    print(f"\n{'─' * 70}")
    print("  Memory Usage")
    print(f"{'─' * 70}")

    for label, factory, _ in configs[:2]:  # Just 3-rxn and 6-rxn at 850C
        peak = bench_memory(factory)
        print(f"  {label}: {peak / 1024:.1f} KB peak")

    # RMG-py memory estimate
    print()
    print("  RMG-py estimated memory (per-object overhead):")
    # RMG Species objects are ~2-5 KB each (molecular graph, thermo, transport)
    # RMG Reaction objects are ~1-3 KB each (kinetics, pairs, degeneracy)
    for label, factory, T in configs[:2]:
        net = factory(T)
        rmg_sp, rmg_rxn = rmg_equivalent_counts(net)
        sp_mem = rmg_sp * 3.5  # ~3.5 KB per Species
        rxn_mem = rmg_rxn * 2.0  # ~2 KB per Reaction
        total_kb = sp_mem + rxn_mem
        print(f"  {label}: ~{total_kb:.0f} KB "
              f"({rmg_sp} species × 3.5 KB + {rmg_rxn} rxns × 2 KB)")

    # Scaling projection
    print(f"\n{'─' * 70}")
    print("  Scaling Projection: QIRN vs RMG-py")
    print(f"{'─' * 70}")

    print(f"  {'N atoms':>8} | {'QIRN states':>12} | {'RMG species':>12} | "
          f"{'RMG rxns (1 rxn)':>16} | {'RMG rxns (6 rxn)':>16}")
    print(f"  {'─' * 8} | {'─' * 12} | {'─' * 12} | {'─' * 16} | {'─' * 16}")
    for n in [3, 5, 8, 10, 13, 15, 20]:
        qirn_states = 2**n
        rmg_species = 2**n  # same count
        rmg_rxns_1 = 2**n * 2**n  # one bimolecular reaction
        rmg_rxns_6 = rmg_rxns_1 * 6  # six reactions
        print(f"  {n:>8} | {qirn_states:>12,} | {rmg_species:>12,} | "
              f"{rmg_rxns_1:>16,} | {rmg_rxns_6:>16,}")

    print()
    print("  Key insight: QIRN always has len(reactions) Reaction objects.")
    print("  RMG-py creates 2^(N_a+N_b) Reaction objects per bimolecular reaction.")
    print("  At N=13 (glucose), one RMG bimolecular reaction becomes 67M objects.")
    print("=" * 70)


if __name__ == "__main__":
    main()
