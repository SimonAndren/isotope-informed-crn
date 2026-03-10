"""
Full RMG isotopologue pipeline for propane pyrolysis.

Stages:
  1. Base mechanism  — RMG generates ~31 unlabeled species / ~150 reactions.
                       Expected time: < 1 min (RMG 3.3.0).
  2. Isotopologue expansion — re-runs RMG reaction enumeration on all ¹³C-labeled
                       variants (~343 species / ~7096 reactions).
                       Expected time: ~44 core-hours (Goldman 2019).
     2b. Cluster CSV — generates isotopomer_cluster_info.csv required for δ¹³C analysis.
  3. YAML conversion + Cantera validation — converts the expanded CHEMKIN file
                       to Cantera YAML and runs a quick simulation to verify the
                       mechanism loaded and propane cracks at 850 °C.
                       Expected time: ~1–5 minutes.

Usage (must use the rmg_env conda environment for stages 1–2):

    conda run --prefix /opt/homebrew/Caskroom/miniforge/base/envs/rmg_env \\
        python rmg_pipeline/run_pipeline.py [--stage {base,expand,simulate,all}]

Stage 3 can be run in the main uv environment:

    uv run python rmg_pipeline/run_pipeline.py --stage simulate

Output is written to rmg_pipeline/output/.

The Goldman 2019 pre-generated mechanisms are in:
    rmg_inputs/goldmanm-RMG_isotopes_paper_data-234bd52/mechanisms/

This script generates the mechanism from scratch so the two can be compared.

Crash recovery:
    Stage 2 has no internal checkpointing. If it crashes, the partial iso/
    directory is detected and removed before a clean restart.
    Completion is marked by the presence of iso/chemkin/chem_annotated.inp.
"""

import argparse
import logging
import os
import shutil
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

# Stage 2 writes into ISO_DIR/iso/ (rmgpy.tools.isotopes.run creates this subdir)
ISO_CHEMKIN = os.path.join(ISO_DIR, "iso", "chemkin", "chem_annotated.inp")
ISO_YAML    = os.path.join(ISO_DIR, "iso", "chem.yaml")


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

    n_species   = len(rmg.reaction_model.core.species)
    n_reactions = len(rmg.reaction_model.core.reactions)
    print(f"\nStage 1 done in {elapsed:.1f}s ({elapsed/3600:.3f} core-hours)")
    print(f"  Species:   {n_species}")
    print(f"  Reactions: {n_reactions}")
    return elapsed


def _stage2_complete() -> bool:
    """Return True if Stage 2 produced a final CHEMKIN file."""
    return os.path.exists(ISO_CHEMKIN)


def _stage2_partial() -> bool:
    """Return True if Stage 2 started but did not finish."""
    iso_subdir = os.path.join(ISO_DIR, "iso")
    return os.path.isdir(iso_subdir) and not _stage2_complete()


def generate_cluster_csv() -> str:
    """Stage 2b: generate isotopomer_cluster_info.csv from expanded CHEMKIN.

    Ports Goldman's prepare_model.py to pandas 2.x (replaces df.append with
    pd.concat). Writes the CSV alongside the CHEMKIN file.

    Returns the path to the written CSV.
    """
    import pandas as pd
    from rmgpy.chemkin import load_chemkin_file
    from rmgpy.tools.isotopes import cluster

    chemkin_file = ISO_CHEMKIN
    dict_file    = os.path.join(ISO_DIR, "iso", "chemkin", "species_dictionary.txt")
    csv_path     = os.path.join(ISO_DIR, "iso", "isotopomer_cluster_info.csv")

    print(f"\n  Loading expanded CHEMKIN for cluster CSV…")
    species, _ = load_chemkin_file(
        chemkin_file, dict_file, read_comments=False, use_chemkin_names=True
    )

    spec_clusters = cluster(species)

    rows = []
    index = []
    for cluster_num, spec_list in enumerate(spec_clusters):
        for spc in spec_list:
            info = {
                "cluster_number":  cluster_num,
                "enriched_atoms":  0,
                "unenriched_atoms": 0,
            }
            for ct in range(5):
                info[f"{ct}_enriched"]   = 0
                info[f"{ct}_unenriched"] = 0
            info["r_enriched"]   = 0
            info["r_unenriched"] = 0

            for atom in spc.molecule[0].atoms:
                if atom.symbol != "C":
                    continue
                n_c_neighbors = sum(
                    1 for nb in atom.bonds if nb.symbol == "C"
                )
                if atom.element.isotope == -1:
                    info["unenriched_atoms"] += 1
                    info[f"{n_c_neighbors}_unenriched"] += 1
                    if atom.radical_electrons > 0:
                        info["r_unenriched"] += 1
                else:
                    info["enriched_atoms"] += 1
                    info[f"{n_c_neighbors}_enriched"] += 1
                    if atom.radical_electrons > 0:
                        info["r_enriched"] += 1

            rows.append(info)
            index.append(spc.label)

    df = pd.DataFrame(rows, index=index)
    df.index.name = "name"

    # Position-specific columns expected by get_psie
    df["not_r_enriched"]   = df["enriched_atoms"]   - df["r_enriched"]
    df["not_r_unenriched"] = df["unenriched_atoms"]  - df["r_unenriched"]

    # Rename 0_* → match Goldman's column names (1=primary,2=secondary,…)
    # Goldman used 1-4 for primary-quaternary; we compute 0-4 (0 = no C neighbours = isolated)
    # Keep all columns — downstream code selects by name.
    df.to_csv(csv_path)
    print(f"  Cluster CSV written: {csv_path} ({len(df)} species, {df['cluster_number'].nunique()} clusters)")
    return csv_path


def run_expand(yes: bool = False, force: bool = False, kie_method: str = "simple_fixed",
               use_original_reactions: bool = True):
    """Stage 2: isotopologue expansion — the expensive step.

    rmgpy.tools.isotopes.run() creates two subdirectories inside ISO_DIR:
      ISO_DIR/iso/ — full isotopologue-expanded mechanism (CHEMKIN format)

    The expanded CHEMKIN file is at ISO_DIR/iso/chemkin/chem_annotated.inp.

    Crash recovery:
      - If Stage 2 already completed, skip immediately (unless force=True).
      - If a partial iso/ directory exists (crash), ask before removing it.

    Args:
        yes:        Auto-confirm removal of partial output.
        force:      Remove completed iso/ and re-run from scratch (use when
                    changing KIE method or other pipeline parameters).
        kie_method: KIE application method: 'simple_fixed' (default, copies
                    base b/Ea before applying KIE factor), 'none' (copies base
                    kinetics but no KIE factor — use to test carbon routing),
                    or 'simple' (original, may leave wrong b/Ea on tree estimates).
        use_original_reactions: If True, map isotopologue reactions from the base
                    mechanism's TemplateReactions (correct degeneracy scaling via
                    new_deg/old_deg ratio). If False, re-enumerate all reactions
                    from scratch via enlarge() — suffers from degeneracy double-
                    counting when ¹³C breaks molecular symmetry.
    """
    from rmgpy.tools.isotopes import run as run_isotopes
    from rmgpy.rmg.main import initialize_log

    iso_subdir = os.path.join(ISO_DIR, "iso")

    # ── Force re-run: wipe completed output ───────────────────────────────
    if force and _stage2_complete():
        print(f"\nForce flag set — removing completed {iso_subdir} for re-run.")
        shutil.rmtree(iso_subdir)
        # Also remove inline-generated base if present (from use_original_reactions=True)
        rmg_subdir = os.path.join(ISO_DIR, "rmg")
        if os.path.isdir(rmg_subdir):
            shutil.rmtree(rmg_subdir)

    # ── Check if already complete ──────────────────────────────────────────
    if _stage2_complete():
        print("\nStage 2 already complete (iso/chemkin/chem_annotated.inp exists). Skipping.")
        csv = os.path.join(ISO_DIR, "iso", "isotopomer_cluster_info.csv")
        if not os.path.exists(csv):
            print("  Cluster CSV missing — regenerating…")
            generate_cluster_csv()
        return 0.0

    # ── Handle partial run ─────────────────────────────────────────────────
    if _stage2_partial():
        print(f"\nWARNING: Partial Stage 2 run detected at {iso_subdir}")
        print("  (iso/ directory exists but chem_annotated.inp is absent)")
        if not yes:
            answer = input("  Remove incomplete iso/ and restart Stage 2? [y/N] ").strip().lower()
            if answer != "y":
                print("  Aborting. Remove iso/ manually to restart.")
                sys.exit(1)
        shutil.rmtree(iso_subdir)
        rmg_subdir = os.path.join(ISO_DIR, "rmg")
        if os.path.isdir(rmg_subdir):
            shutil.rmtree(rmg_subdir)
        print(f"  Removed {iso_subdir}")

    os.makedirs(ISO_DIR, exist_ok=True)
    initialize_log(logging.INFO, os.path.join(ISO_DIR, "RMG.log"))

    # When use_original_reactions=True, generate base inline so reactions
    # retain TemplateReaction.template attributes (lost when loading from
    # saved CHEMKIN — tree-estimated reactions lack "rate rule" comments).
    original_dir = None if use_original_reactions else BASE_DIR
    base_label = "inline (fresh Stage 1)" if original_dir is None else BASE_DIR

    print(f"\n{'='*60}")
    print("Stage 2: Isotopologue expansion")
    print(f"Using base from: {base_label}")
    print(f"Output: {ISO_DIR}")
    print(f"KIE method: {kie_method}")
    print(f"use_original_reactions: {use_original_reactions}")
    print("WARNING: This step takes ~44 core-hours (Goldman 2019).")
    print("="*60)

    t0 = time.perf_counter()
    kie_arg = None if kie_method == "disabled" else kie_method
    run_isotopes(
        input_file=INPUT_FILE,
        output_directory=ISO_DIR,
        original=original_dir,
        maximum_isotopic_atoms=1000000,
        use_original_reactions=use_original_reactions,
        kinetic_isotope_effect=kie_arg,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nStage 2 done in {elapsed:.1f}s ({elapsed/3600:.3f} core-hours)")

    print("\nStage 2b: Generating isotopomer_cluster_info.csv…")
    generate_cluster_csv()

    return elapsed


def run_simulate(yaml_path: str = ISO_YAML):
    """Stage 3: YAML conversion + Cantera validation.

    1. Converts the expanded CHEMKIN mechanism to Cantera YAML format.
    2. Loads the YAML with Cantera and runs a quick simulation at 850 °C
       to verify the mechanism works end-to-end.

    Args:
        yaml_path: Path to the YAML file to produce (and then simulate).
                   Defaults to ISO_YAML (generated mechanism); pass a Goldman
                   pre-generated path to skip the conversion step.
    """
    import cantera as ct
    import cantera.ck2yaml as ck2yaml

    print(f"\n{'='*60}")
    print("Stage 3: YAML conversion + Cantera validation")

    # If the YAML doesn't already exist, convert from CHEMKIN.
    if not os.path.exists(yaml_path):
        if not os.path.exists(ISO_CHEMKIN):
            print(f"ERROR: Expanded mechanism not found: {ISO_CHEMKIN}")
            print("Run --stage expand first.")
            sys.exit(1)

        print(f"Converting: {ISO_CHEMKIN}")
        print(f"       → {yaml_path}")
        ck2yaml.convert(input_file=ISO_CHEMKIN, out_name=yaml_path, quiet=True)

    print(f"Loading: {yaml_path}")
    gas = ct.Solution(yaml_path)
    print(f"  Loaded: {gas.n_species} species, {gas.n_reactions} reactions")

    # Quick validation: simulate propane pyrolysis at 850 °C for 85 ms.
    T = 1123.0   # 850 °C in K
    P = 2.0e5    # 2 bar in Pa

    # Identify unlabeled propane: either SMILES "CCC" or RMG label "propane_ooo(N)".
    propane = next(
        (s for s in gas.species_names if s == "CCC" or s.startswith("propane_ooo")),
        None,
    )
    if propane is None:
        print("WARNING: propane not found in mechanism; skipping simulation.")
        return gas

    he = next((s for s in gas.species_names if s in ("[He]", "He")), None)
    init_x = {propane: 0.01}
    if he:
        init_x[he] = 0.99

    gas.TPX = T, P, init_x
    r = ct.IdealGasConstPressureReactor(gas, energy="off")
    sim = ct.ReactorNet([r])

    t0 = time.perf_counter()
    sim.advance(0.085)
    elapsed = time.perf_counter() - t0

    propane_idx = gas.species_index(propane)
    x_propane_final = r.phase.X[propane_idx]
    conversion = max(0.0, 1.0 - x_propane_final / 0.01)

    print(f"\nCantera simulation at {T:.0f} K (850 °C), 85 ms:")
    print(f"  Propane conversion: {conversion*100:.1f}%")
    print(f"  Simulation time:    {elapsed:.2f}s")
    print("="*60)
    return gas


def main():
    parser = argparse.ArgumentParser(description="RMG propane isotopologue pipeline")
    parser.add_argument(
        "--stage",
        choices=["base", "expand", "simulate", "all"],
        default="base",
        help="Which stage to run (default: base only)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-confirm removal of partial Stage 2 output (for background runs)",
    )
    parser.add_argument(
        "--force-expand",
        action="store_true",
        help="Remove completed Stage 2 output and re-run from scratch "
             "(use when changing KIE method or pipeline parameters)",
    )
    parser.add_argument(
        "--kie-method",
        choices=["simple_corrected", "simple_fixed", "none", "simple", "disabled"],
        default="simple_corrected",
        help="KIE application method for Stage 2: 'simple_corrected' (default, applies "
             "reduced-mass KIE on existing kinetics — preserves degeneracy correction "
             "from use_original_reactions), 'simple_fixed' (copies base b/Ea before KIE "
             "factor — use with enlarge()), 'none' (copies base kinetics, no KIE factor), "
             "'simple' (original, may have wrong b/Ea), 'disabled' (skip KIE entirely).",
    )
    parser.add_argument(
        "--use-original-reactions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stage 2 reaction generation: --use-original-reactions (default) maps "
             "isotopologue reactions from the base mechanism's TemplateReactions with "
             "correct degeneracy scaling (new_deg/old_deg). --no-use-original-reactions "
             "re-enumerates all reactions via enlarge() (suffers from degeneracy double-"
             "counting when 13C breaks molecular symmetry).",
    )
    args = parser.parse_args()

    if args.stage in ("base", "all"):
        run_base()
    if args.stage in ("expand", "all"):
        if not os.path.exists(os.path.join(BASE_DIR, "chemkin", "chem_annotated.inp")):
            print("ERROR: Run --stage base first to generate the base mechanism.")
            sys.exit(1)
        run_expand(yes=args.yes, force=args.force_expand, kie_method=args.kie_method,
                   use_original_reactions=args.use_original_reactions)
    if args.stage in ("simulate", "all"):
        run_simulate()

    print("\nDone.")


if __name__ == "__main__":
    main()
