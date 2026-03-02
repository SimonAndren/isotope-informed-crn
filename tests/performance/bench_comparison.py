#!/usr/bin/env python
"""Head-to-head benchmark: RMG-Py isotopologue ODE vs vector engine.

Runs the same propane pyrolysis network (3-rxn and 6-rxn) through both
approaches and compares wall time, RHS call count, and memory.

Run with the rmg_env Python interpreter for the RMG side:
    /opt/homebrew/Caskroom/miniforge/base/envs/rmg_env/bin/python \\
        tests/performance/bench_comparison.py

Or with uv for vector-only results (RMG gracefully skipped):
    uv run python tests/performance/bench_comparison.py

Results saved to comparison_result.json.
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc

import numpy as np

# ── Project imports (always available) ───────────────────────────────────────
sys.path.insert(0, "src")
from isotopologue.benchmarks.propane import initial_conditions, propane_3rxn, propane_6rxn
from isotopologue.engine import IsotopologueEngine

# ── RMG-Py imports (optional) ────────────────────────────────────────────────
sys.path.insert(0, "vendor/RMG-Py")
try:
    from tests.performance.bench_rmg_isotope import (  # noqa: E402
        bench_rmg_ode,
        make_propane_species,
    )

    HAS_RMG = True
except ImportError:
    # Try loading bench_rmg_isotope directly if the module path differs
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "bench_rmg_isotope",
            "tests/performance/bench_rmg_isotope.py",
        )
        mod = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(mod)
        bench_rmg_ode = mod.bench_rmg_ode
        make_propane_species = mod.make_propane_species
        HAS_RMG = mod.HAS_RMG
    except Exception as e:
        HAS_RMG = False
        _RMG_ERR = str(e)


T = 1123.0   # K — 850°C, supplement conditions
T_END = 0.085  # s — 85 ms residence time


# ─── Vector engine helpers ────────────────────────────────────────────────────

def _bench_vector_rhs(engine: IsotopologueEngine, y0: np.ndarray, n: int = 10_000) -> float:
    """Return µs per RHS call (warmed up)."""
    for _ in range(200):
        engine.rhs(0.0, y0)
    t0 = time.perf_counter()
    for _ in range(n):
        engine.rhs(0.0, y0)
    return (time.perf_counter() - t0) / n * 1e6


def _bench_vector_solve(engine: IsotopologueEngine, y0: np.ndarray, t_end: float):
    """Return (wall_ms, n_rhs_calls, success)."""
    calls = [0]
    orig = engine.rhs

    def counting_rhs(t, y):
        calls[0] += 1
        return orig(t, y)

    engine.rhs = counting_rhs
    t0 = time.perf_counter()
    result = engine.solve(y0, (0, t_end), method="BDF", rtol=1e-8, atol=1e-12)
    elapsed = (time.perf_counter() - t0) * 1000
    engine.rhs = orig
    return elapsed, calls[0], result.success


def _bench_vector_memory(factory, t_end: float = T_END) -> float:
    """Peak memory (KB) for vector engine setup + solve."""
    tracemalloc.start()
    net = factory(T)
    engine = IsotopologueEngine(network=net)
    conc = initial_conditions(net)
    y0 = net.pack(conc)
    engine.solve(y0, (0, t_end), method="BDF", rtol=1e-8, atol=1e-12)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024


def _count_state_variables(net):
    return sum(sp.n_isotopologues for sp in net.species.values())


def _rmg_equiv_counts(net):
    n_species = sum(sp.n_isotopologues for sp in net.species.values())
    n_reactions = 0
    for rxn in net.reactions:
        if rxn.reaction_type in ("synthesis", "exchange"):
            n_reactions += (
                net.species[rxn.reactants[0]].n_isotopologues
                * net.species[rxn.reactants[1]].n_isotopologues
            )
        else:
            n_reactions += net.species[rxn.reactants[0]].n_isotopologues
    return n_species, n_reactions


# ─── Print helpers ────────────────────────────────────────────────────────────

def _bar(label: str, value: str, width: int = 28):
    return f"  {label:<{width}} {value}"


def _print_table(rows: list[tuple[str, str]]):
    for label, val in rows:
        print(_bar(label, val))


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_comparison(factory, label: str, n_rxn_model: str) -> dict:
    print(f"\n{'═' * 70}")
    print(f"  {label}")
    print(f"{'═' * 70}")

    # ── Vector engine ─────────────────────────────────────────────────────────
    net = factory(T)
    engine = IsotopologueEngine(network=net)
    conc = initial_conditions(net)
    y0 = net.pack(conc)

    n_state = _count_state_variables(net)
    rmg_sp_equiv, rmg_rxn_equiv = _rmg_equiv_counts(net)

    t0 = time.perf_counter()
    _ = IsotopologueEngine(network=net)
    vec_setup_ms = (time.perf_counter() - t0) * 1000

    rhs_us = _bench_vector_rhs(engine, y0)
    vec_solve_ms, vec_rhs_calls, vec_ok = _bench_vector_solve(engine, y0, T_END)
    vec_mem_kb = _bench_vector_memory(factory)

    print(f"\n  {'─' * 30}  Vector Engine  {'─' * 22}")
    _print_table([
        ("Species in network:", str(len(net.species))),
        ("Reactions in network:", str(len(net.reactions))),
        ("State variables:", str(n_state)),
        ("State vector bytes:", str(y0.nbytes)),
        ("Setup time:", f"< {vec_setup_ms:.3f} ms"),
        ("Single RHS:", f"{rhs_us:.2f} µs"),
        ("85ms solve time:", f"{vec_solve_ms:.2f} ms  ({vec_rhs_calls} RHS calls)"),
        ("Peak memory:", f"{vec_mem_kb:.1f} KB"),
        ("Solve succeeded:", str(vec_ok)),
    ])

    # ── RMG-Py ────────────────────────────────────────────────────────────────
    rmg_result = None
    if HAS_RMG:
        species_dict = make_propane_species()
        print(f"\n  {'─' * 30}  RMG-Py         {'─' * 22}")
        rmg_result = bench_rmg_ode(
            species_dict, max_isotopes=3, T=T, n_rxn_model=n_rxn_model
        )
        _print_table([
            ("Isotopologue species:", str(rmg_result["n_isotopologue_species"])),
            ("Isotopologue reactions:", str(rmg_result["n_isotopologue_reactions"])),
            ("Isotopomer gen:", f"{rmg_result['isotopomer_gen_ms']:.2f} ms"),
            ("Reaction generation:", f"{rmg_result['reaction_gen_ms']:.2f} ms"),
            ("ODE setup:", f"{rmg_result['ode_setup_ms']:.2f} ms"),
            ("ODE solve (85ms sim):", f"{rmg_result['ode_solve_ms']:.2f} ms"),
            ("Total pipeline:", f"{rmg_result['ode_total_ms']:.2f} ms"),
        ])

        # ── Speedup ──────────────────────────────────────────────────────────
        speedup_solve = rmg_result["ode_solve_ms"] / vec_solve_ms if vec_solve_ms > 0 else float("inf")
        speedup_total = rmg_result["ode_total_ms"] / vec_solve_ms if vec_solve_ms > 0 else float("inf")
        print(f"\n  {'─' * 30}  Speedup         {'─' * 22}")
        _print_table([
            ("ODE solve only:", f"{speedup_solve:.1f}×  (RMG/Vector)"),
            ("Full pipeline vs solve:", f"{speedup_total:.1f}×"),
            ("RMG equiv species:", str(rmg_sp_equiv)),
            ("RMG equiv reactions:", str(rmg_rxn_equiv)),
            ("Rxn compression:", f"{rmg_rxn_equiv / len(net.reactions):.1f}× fewer Reaction objects"),
        ])
    else:
        print(f"\n  RMG-Py: not available (run with rmg_env Python)")
        _print_table([
            ("RMG-equiv species:", str(rmg_sp_equiv)),
            ("RMG-equiv reactions:", str(rmg_rxn_equiv)),
            ("Reaction compression:", f"{rmg_rxn_equiv / len(net.reactions):.1f}×"),
        ])

    return {
        "label": label,
        "vector": {
            "n_species": len(net.species),
            "n_reactions": len(net.reactions),
            "n_state": n_state,
            "setup_ms": vec_setup_ms,
            "rhs_us": rhs_us,
            "solve_ms": vec_solve_ms,
            "rhs_calls": vec_rhs_calls,
            "memory_kb": vec_mem_kb,
            "success": vec_ok,
        },
        "rmg": rmg_result,
        "rmg_equiv_species": rmg_sp_equiv,
        "rmg_equiv_reactions": rmg_rxn_equiv,
    }


def main():
    print("=" * 70)
    print("  Isotopologue ODE: RMG-Py vs Vector Engine — Head-to-Head")
    print("=" * 70)
    print(f"\n  Conditions: T={T} K (850°C), P=2 bar, propane pyrolysis, t=85 ms")
    print(f"  RMG-Py available: {HAS_RMG}")

    results = []
    configs = [
        (propane_3rxn, "Propane 3-rxn @ 850°C", "3rxn"),
        (propane_6rxn, "Propane 6-rxn @ 850°C", "6rxn"),
        (lambda _=None: propane_3rxn(1223.0), "Propane 3-rxn @ 950°C", "3rxn"),
        (lambda _=None: propane_6rxn(1223.0), "Propane 6-rxn @ 950°C", "6rxn"),
    ]

    for factory, label, n_rxn_model in configs:
        r = run_comparison(factory, label, n_rxn_model)
        results.append(r)

    # ── Scaling projection ────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  Scaling: state variables and explicit reactions vs molecule size")
    print(f"{'═' * 70}")
    header = f"  {'N atoms':>8} | {'States':>8} | {'RMG rxns (1 bimol)':>20} | {'RMG rxns (6 rxn)':>18}"
    print(header)
    print(f"  {'─' * 8} | {'─' * 8} | {'─' * 20} | {'─' * 18}")
    for n in [3, 5, 6, 8, 10, 13]:
        states = 2 ** n
        rmg_1 = 2 ** (2 * n)
        rmg_6 = rmg_1 * 6
        print(f"  {n:>8} | {states:>8,} | {rmg_1:>20,} | {rmg_6:>18,}")

    print(f"\n  Key insight: vector engine has exactly len(reactions) Reaction objects.")
    print(f"  RMG-Py creates 2^(nA+nB) explicit reactions per bimolecular pair.")

    # ── Save results ──────────────────────────────────────────────────────────
    with open("comparison_result.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("Results saved to comparison_result.json")


if __name__ == "__main__":
    main()
