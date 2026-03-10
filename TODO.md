# Project TODO

Items are roughly ordered by priority within each section. Add freely — delete
when done.

---

## Completed: RMG Pipeline — Delta-13C Fix

- [x] Simulate Goldman's full_model YAML and confirm Table 2 = 0.5.
- [x] Identify root cause of delta-CH4 = +462 permil (degeneracy double-counting
      in `enlarge()` when `use_original_reactions=False`).
- [x] Fix via `use_original_reactions=True` + `simple_corrected` KIE method.
      Table 2: 7.9 -> 1.2. Delta-CH4 at 850C: +150 -> -28.4 permil.
- [x] Validate analysis pipeline against Goldman's DRG mechanism (Exp 1).
- [x] Fix `cluster[-1]` assumption with `_find_unlabeled_reaction()` helper.
- [x] Run compare_to_goldman.py — Table 2 = 1.2 (Goldman full = 0.5, DRG = 0.6).

See `docs/delta13c_investigation.md` for the full investigation summary.

---

## Next: Close the Table 2 Gap (1.2 -> 0.5)

- [ ] **Automate species annotation** — `OUR_SPECIES_MAP` in
      `compare_to_goldman.py` is manually constructed from
      `species_dictionary.txt`. Species indices change between Stage 2 runs.
      Write a function that reads cluster CSV + species dictionary and builds
      the map automatically.

- [ ] **Understand entropy-only isotope effect** — With KIE disabled (run-8),
      there is a ~2.5 permil offset from Goldman. This comes from
      `correct_entropy()` in RMG's isotopes.py. Determine whether Goldman's
      mechanism includes this correction and how it interacts with KIE.

- [ ] **Investigate dC2H4 slope deviation** — Our dC2H4 = f(dCH4) slope is
      systematically 0.02-0.03 higher than Goldman at all temperatures. This is
      the main contributor to the Table 2 gap. Likely a kinetics database
      difference (RMG 3.3.0 vs 2.4.1). Options:
      - Compare specific rate constants for C2H4-forming reactions
      - Try seeding Goldman's base kinetics into our Stage 2

- [ ] **Match Goldman's species count** — Our mechanism has 187 isotopologue
      species vs Goldman's 343. Missing C5/C6 species are in our RMG edge but
      not promoted to core. Lower `toleranceMoveToCore` or seed species.

---

## Next: QIRN Vectorized Approach

- [ ] **Implement QIRN-style isotope tracking** — Carry isotope fractions as a
      vector alongside the base mechanism ODE (Mueller & Wu 2022). Avoids the
      combinatorial explosion of isotopologue species.
      - Start with the 3-rxn propane model as a test case
      - Validate against Goldman's 3-rxn mechanism delta values
      - Compare results to our RMG-expanded mechanism (run-9)

- [ ] **Write QIRN integration tests** — Tests that verify QIRN and RMG
      approaches produce the same delta-13C slopes when given the same base
      mechanism and conditions.

- [ ] **Connect QIRN reference code** — `vendor/QIRN-Files/Scripts/QIRN.py`
      implements the Caltech approach. Run it on propane and verify it matches
      our engine output.

---

## Science: Replication & Validation

- [x] Run Goldman pipeline from scratch; compare to full_model.
- [ ] Implement atom maps for all exchange reactions in `benchmarks/propane.py`:
      - R2: CH3 + C2H5 -> CH4 + C2H4 (disproportionation)
      - R5: C2H6 + CH3 -> CH4 + C2H5 (H-abstraction)
      - R6: C2H6 + H -> H2 + C2H5 (H-abstraction)
- [ ] Add site-specific KIEs to propane benchmark reactions.
- [ ] Tighten `test_gilbert_replication.py` tolerances once atom maps + KIEs
      are in place — target enrichment slopes within 2-sigma of Gilbert Table 2.
- [ ] Add full 18-reaction DRG propane benchmark (match Goldman DRG model).

---

## Performance

- [ ] Profile `_apply_exchange` on the 6-rxn network.
- [ ] Performance test: bench_temperatures.py for 800-950C sweep.

---

## Infrastructure

- [ ] Add GitHub Actions CI: `uv run pytest tests/unit/` on push.
- [ ] Add `ruff` and `mypy` to CI checks.
- [ ] Write a Makefile or `justfile` with common commands.
