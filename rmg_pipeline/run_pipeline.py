"""
Full RMG isotopologue pipeline for propane pyrolysis.

Stages:
  1. Base mechanism  — RMG generates ~20 unlabeled species / ~30 reactions.
                       Expected time: ~5–30 minutes.
  2. Isotopologue expansion — re-runs RMG reaction enumeration on all ¹³C-labeled
                       variants (~343 species / ~7096 reactions).
                       Expected time: ~44 core-hours (Goldman 2019).

Usage (must use the rmg_env conda environment):

    conda run --prefix /opt/homebrew/Caskroom/miniforge/base/envs/rmg_env \\
        python rmg_pipeline/run_pipeline.py [--stage {base,expand,all}]

Output is written to rmg_pipeline/output/.

The Goldman 2019 pre-generated mechanisms are in:
    rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52/mechanisms/

This script generates the mechanism from scratch so the two can be compared.
"""

import argparse
import logging
import os
import sys
import time

RMG_PY = os.path.abspath("vendor/RMG-Py")
sys.path.insert(0, RMG_PY)

INPUT_FILE = os.path.abspath(
    "rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52/input_files/propane_rmg_input_file.py"
)
OUTPUT_DIR = os.path.abspath("rmg_pipeline/output")
BASE_DIR   = os.path.join(OUTPUT_DIR, "base")
ISO_DIR    = os.path.join(OUTPUT_DIR, "isotope")


def run_base():
    """Stage 1: generate unlabeled propane mechanism with RMG."""
    from rmgpy.rmg.main import RMG, initialize_log

    os.makedirs(BASE_DIR, exist_ok=True)
    initialize_log(logging.INFO, os.path.join(BASE_DIR, "RMG.log"))

    print(f"\n{'='*60}")
    print("Stage 1: Base mechanism generation")
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {BASE_DIR}")
    print("Expected time: 5–30 minutes")
    print("="*60)

    t0 = time.perf_counter()
    rmg = RMG(input_file=INPUT_FILE, output_directory=BASE_DIR)
    rmg.execute()
    elapsed = time.perf_counter() - t0

    n_species  = len(rmg.reaction_model.core.species)
    n_reactions = len(rmg.reaction_model.core.reactions)
    print(f"\nStage 1 done in {elapsed:.1f}s ({elapsed/3600:.3f} core-hours)")
    print(f"  Species:   {n_species}")
    print(f"  Reactions: {n_reactions}")
    return elapsed


def run_expand():
    """Stage 2: isotopologue expansion — the expensive step."""
    from rmgpy.tools.isotopes import run as run_isotopes
    from rmgpy.rmg.main import initialize_log

    os.makedirs(ISO_DIR, exist_ok=True)
    initialize_log(logging.INFO, os.path.join(ISO_DIR, "RMG.log"))

    print(f"\n{'='*60}")
    print("Stage 2: Isotopologue expansion")
    print(f"Using base from: {BASE_DIR}")
    print(f"Output: {ISO_DIR}")
    print("WARNING: This step takes ~44 core-hours (Goldman 2019).")
    print("="*60)

    t0 = time.perf_counter()
    run_isotopes(
        input_file=INPUT_FILE,
        output_directory=ISO_DIR,
        original=BASE_DIR,              # skip re-running base
        maximum_isotopic_atoms=1000000,
        use_original_reactions=False,   # full re-enumeration (matches Goldman)
        kinetic_isotope_effect="simple",
    )
    elapsed = time.perf_counter() - t0
    print(f"\nStage 2 done in {elapsed:.1f}s ({elapsed/3600:.3f} core-hours)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="RMG propane isotopologue pipeline")
    parser.add_argument(
        "--stage", choices=["base", "expand", "all"], default="base",
        help="Which stage to run (default: base only)"
    )
    args = parser.parse_args()

    if args.stage in ("base", "all"):
        t_base = run_base()
    if args.stage in ("expand", "all"):
        if not os.path.exists(os.path.join(BASE_DIR, "chemkin", "chem_annotated.inp")):
            print("ERROR: Run --stage base first to generate the base mechanism.")
            sys.exit(1)
        t_expand = run_expand()

    print("\nDone.")


if __name__ == "__main__":
    main()
