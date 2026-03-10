# Experiment Log

Append-only. Read this before running any pipeline stage.

---

## 2026-03-05 run-1 — Stage 2 `simple` (CRASH)

**Command:** Stage 2 with default `kinetic_isotope_effect='simple'`
**KIE method:** `simple`
**Outcome:** CRASH
**Error:** `ActionError: Could not find labeled reactants for reaction [CH2]CC(23) <=> [H](22) + C3H6(45) from family R_Addition_MultipleBond`
**Conclusion:** `simple` method crashes on R_Addition_MultipleBond clusters where cluster[-1] is the unimolecular reverse reaction. Do not use `simple`.

---

## 2026-03-05 run-2 — Stage 2 `simple_fixed`

**Command:** Stage 2 with `kinetic_isotope_effect='simple_fixed'`
**KIE method:** `simple_fixed`
**Species/reactions:** 187 species / 5086 reactions (our network); Goldman has 343/7096
**Outcome:** SUCCESS → `rmg_pipeline/output_backup/isotope/iso/chem.yaml`
**Stage 3:** Cantera simulation at 850°C, 85ms → ~99% propane conversion
**Table 2 score:** Ours = 7.9 (target ≤ 0.6); Goldman full_model = 0.5
**δ¹³C(CH4):** Ours ≈ +150‰; Goldman ≈ −21‰
**Conclusion:** `simple_fixed` runs without crashing and CH3 recombination has correct b/Ea.
But Table 2 score is 7.9, far from target. Root cause unknown — see `docs/kie_diagnosis.md`.

---

## 2026-03-06 run-3 — Stage 2 `none` attempt 1 (CRASH)

**Command:** `python rmg_pipeline/run_pipeline.py --stage expand --kie-method none --force-expand --yes`
**KIE method:** `none` (first implementation, no try/except, no family filter)
**Outcome:** CRASH
**Error:** `AssertionError: Gas phase reaction [H](22)+C3H6(46)<=>C3H7(24) with kinetics Arrhenius(A=(9.95e+30,'s^-1')...) with 2 reactants was expected to have kinetics.A.get_conversion_factor_from_si_to_cm_mol_s() = 1000000.0 but instead it is 1.0`
**Root cause:** `apply_kinetic_isotope_effect_none` deepcopied `cluster[-1]` kinetics (unimolecular s⁻¹) onto a bimolecular labeled reaction. CHEMKIN writer asserts unit match.
**Conclusion:** Need guard before deepcopy. `r_addition_multiplebond` is problematic.

---

## 2026-03-06 run-4 — Stage 2 `none` attempt 2 (CRASH)

**Command:** same as run-3 after fix attempt (added try/except + supported_families filter but kept `r_addition_multiplebond` in the set)
**KIE method:** `none` (second implementation)
**Outcome:** CRASH — same AssertionError as run-3
**Root cause:** try/except wraps the deepcopy loop; deepcopy itself raises no exception (it succeeds). The AssertionError fires in `save_chemkin_file` which runs AFTER the KIE function returns — outside the try/except scope.
**Conclusion:** try/except is useless here. Must prevent the bad copy from happening, not catch it after the fact. Solution: remove `r_addition_multiplebond` from supported_families OR use `get_labeled_reactants` as a guard.

---

## 2026-03-06 run-5 — Stage 2 `none` attempt 3 (IN PROGRESS)

**Command:** `python rmg_pipeline/run_pipeline.py --stage expand --kie-method none --force-expand --yes`
**KIE method:** `none` (third implementation: `r_addition_multiplebond` removed from supported_families; PID 58297)
**Outcome:** Running (~15-20 min from 09:49 local time)
**Note:** Code on disk was subsequently updated to the correct final implementation (Goldman-style: `get_labeled_reactants` guard, no deepcopy, no `change_rate`). Running process uses version 3 (removed from set). Future runs will use the correct Goldman-style version.
**Outcome:** SUCCESS
**Table 2 score:** 8.5 (worse than simple_fixed = 7.9)
**δCH4 range:** +182‰ to +542‰ (800–950°C), cf. Goldman ≈ −28‰
**Fitted slopes (dC2H4 = f(dCH4)):** 0.430–0.491 (Goldman: 0.502–0.576)
**Conclusion:** Labeled reactions with tree-estimated kinetics (no correction) give δCH4 ≈ +430‰
at 850°C. `simple_fixed` improves this to +150‰ by copying base kinetics. The remaining +150‰
error after `simple_fixed` is NOT from KIE magnitude — it is from the mechanism itself
(likely missing C5/C6 species or routing differences vs Goldman's 343-sp network).
**NOTE:** This `none` run is NOT a pure routing test (labeled reactions keep tree-estimated kinetics,
not base kinetics). A pure routing test would require deepcopy of base kinetics with the unit
mismatch fixed. Current result shows tree-estimation error contributes ~+280‰ offset; `simple_fixed`
partially corrects this.

---

## 2026-03-06 run-6 — Stage 2 `none` pure routing test

**Command:** `python rmg_pipeline/run_pipeline.py --stage expand --kie-method none --force-expand --yes`
**KIE method:** `none` v4 (deepcopy base kinetics + get_labeled_reactants guard, no change_rate; PID 63767)
**What changed vs run-5:** Labeled reactions now get exact copy of base kinetics (not tree-estimated).
**Outcome:** SUCCESS → `rmg_pipeline/output/isotope/iso/chem.yaml` (timestamp 15:52 Mar 6)
**Stage 2 log:** 10 KIE warnings (same families as run-2: Singlet_Carbene, 1,3_sigmatropic, R_Addition guard)
**Propane conversion at 850°C:** ~60% (confirmed by Cantera, same as run-2's yaml)
**Table 2 score:** 8.7 (worse than run-2 = 7.9)
**δCH4@850°C:** ≈ +462‰ (run-5 was +430‰, run-2 was +150‰)
**Conclusion:** With identical kinetics for all isotopologues (deepcopy, no KIE), expected δCH4 ≈ −28‰.
Getting +462‰ is physically impossible without a routing error. Something fundamental is wrong
in how our mechanism routes ¹³C through the network. The 0.18% KIE factor in `simple_fixed` cannot
explain a 300‰ difference. Root cause unresolved — see updated `docs/kie_diagnosis.md`.
**Diagnostic (2026-03-06):** Species "He", "CCC(2-6)", "propane_ooo(1)" all confirmed present in yaml.
C(31)=¹²CH4, C(32)=¹³CH4 confirmed correct. Mass balance verified (¹²C-only propane → ¹³CH4=0).
The "99% conversion" figure cited earlier was incorrect — both yamls give ~60% at 850°C.

---

## 2026-03-06 simple-run attempt — Stage 1→2→3 with Goldman's `simple` (FAILED)

**Command:** Chain script targeting full pipeline with `--kie-method simple`
**Outcome:** FAILED — Stage 1 failed due to shell variable expansion issue in chain script
(`conda run --prefix ...` can't be interpolated as a shell variable in `eval` context)
**Result for `simple`:** Never ran. The compare script in the chain reused run-6's yaml → Table 2 = 8.7 (invalid).
**Next step:** Run `simple` properly with fixed shell command.

---

## 2026-03-09 run-7 — Stage 2 `simple_fixed` with cluster[-1] fix (ABANDONED)

**Command:** `--stage expand --kie-method simple_fixed --force-expand --yes` (with `_find_unlabeled_reaction` fix)
**KIE method:** `simple_fixed` (with cluster[-1] fix)
**use_original_reactions:** False
**Outcome:** ABANDONED — superseded by degeneracy root cause analysis (see below).
**Conclusion:** Even with cluster[-1] fix, the `use_original_reactions=False` path has a fundamental degeneracy double-counting bug. The fix for cluster[-1] cannot resolve the +462‰ enrichment on its own.

---

## 2026-03-09 run-8 — Stage 2 `use_original_reactions=True` + no KIE (SUCCESS)

**Command:** `--stage expand --kie-method disabled --use-original-reactions --force-expand --yes`
**KIE method:** disabled (kinetic_isotope_effect=None, skips KIE entirely)
**use_original_reactions:** True (uses `generate_isotope_reactions` with `change_rate(new_deg/old_deg)`)
**Base generation:** Inline (Stage 1 ran inside Stage 2 to preserve TemplateReaction.template attributes)
**Rationale:** Root cause of +462‰ enrichment is degeneracy double-counting in `enlarge()`.
The `use_original_reactions=True` path produces correct degeneracy by scaling from the base
reaction. KIE is disabled to avoid overwriting the degeneracy correction (simple_fixed
deepcopies base kinetics, losing the deg scaling). KIE contributes ~10‰; the bug is +462‰.

**Outcome:** SUCCESS
**Species/reactions:** 187 species / 4750 reactions (vs 5086 with enlarge; 336 fewer = overcounted channels removed)
**Stage 2 time:** 112.8s (vs ~15-17min with enlarge — 10× faster)
**Propane conversion at 850°C:** 49.1% (vs ~60% with enlarge; Goldman = 42.6%)

**Table 2 score:** 1.2 (was 7.9 with enlarge; Goldman full = 0.5)
**δCH4 @ 850°C, psia=0:** ≈ −30.5‰ (was +150‰; Goldman ≈ −28.3‰)

**Fitted slopes:**
```
                 800   850   900   950
dC2H4 = f(dCH4)  0.518 0.548 0.589 0.609   (Goldman: 0.502 0.514 0.545 0.576)
dC2H6 = f(dCH4)  0.996 0.995 0.989 0.985   (Goldman: 0.979 0.977 0.975 0.985)
dC2H6 = f(dC2H4) 1.923 1.815 1.680 1.619   (Goldman: 1.952 1.902 1.788 1.710)
dBulk = f(dCH4)  0.667 0.668 0.673 0.675   (Goldman: 0.666 0.667 0.672 0.680)
```

**Conclusion:** The degeneracy double-counting bug in `enlarge()` was the primary cause of the
+462‰ enrichment error. Switching to `use_original_reactions=True` with correct degeneracy
scaling reduces Table 2 from 7.9 to 1.2 — between Goldman's DRG (0.6) and 3rxn (1.7) models.
The remaining gap (1.2 vs 0.5) is likely from: (1) missing C5/C6 species, (2) no KIE applied.
δ¹³C values are now in the physically correct range (−8‰ to −38‰ vs Goldman's −6‰ to −38‰).

**Note on species naming:** Inline Stage 1 generates different species indices than loading
from saved CHEMKIN. Updated `compare_to_goldman.py` with new species map:
- Methane: C(46)/C(47) cluster 16 (was C(31)/C(32) cluster 19)
- Ethylene: C2H4(14)/C2H4(15)/C2H4(16) cluster 22 (was 33/34/35 cluster 18)
- Ethane: CC(48)/CC(49)/CC(50) cluster 15 (was CC(7)/CC(8)/CC(9) cluster 25)
- Propane: unchanged (propane_ooo(1), CCC(2-6), cluster 26)

---

## 2026-03-09 run-9 — Stage 2 `use_original_reactions=True` + `simple_corrected` KIE (SUCCESS)

**Command:** `--stage expand --kie-method simple_corrected --use-original-reactions --force-expand --yes`
**KIE method:** `simple_corrected` (applies `change_rate(sqrt(μ_base/μ_labeled))` on existing kinetics — preserves degeneracy correction)
**use_original_reactions:** True
**Base generation:** Inline (same as run-8)

**Outcome:** SUCCESS
**Species/reactions:** 187 species / 4750 reactions (identical to run-8)
**Stage 2 time:** 134.3s (slightly longer than run-8 due to KIE computation)
**Propane conversion at 850°C:** 49.1% (unchanged from run-8)

**Table 2 score:** 1.2 (unchanged from run-8; Goldman full = 0.5)

**Fitted slopes:**
```
                 800   850   900   950
dC2H4 = f(dCH4)  0.516 0.546 0.586 0.606   (run-8: 0.518 0.548 0.589 0.609)
dC2H6 = f(dCH4)  0.979 0.978 0.975 0.977   (run-8: 0.996 0.995 0.989 0.985)
dC2H6 = f(dC2H4) 1.896 1.793 1.664 1.612   (run-8: 1.923 1.815 1.680 1.619)
dBulk = f(dCH4)  0.665 0.666 0.671 0.673   (run-8: 0.667 0.668 0.673 0.675)
```

**Comparison to Goldman:**
```
                 800   850   900   950
dC2H4 = f(dCH4)  0.516 0.546 0.586 0.606   (Goldman: 0.502 0.514 0.545 0.576)
dC2H6 = f(dCH4)  0.979 0.978 0.975 0.977   (Goldman: 0.979 0.977 0.975 0.985)
dBulk = f(dCH4)  0.665 0.666 0.671 0.673   (Goldman: 0.666 0.667 0.672 0.680)
```

**Conclusion:** KIE has minimal impact on Table 2 score (still 1.2) because the reduced-mass
correction for ¹³C/¹²C is only ~3.8% change in rate constant, which translates to ~1-2‰ shift
in slopes at pyrolysis temperatures. However, the KIE does bring dC2H6 slopes near-identical to
Goldman (0.975-0.979 vs 0.975-0.985). The remaining gap (1.2 vs 0.5) is primarily in the dC2H4
slope, which is systematically ~0.02-0.03 higher than Goldman at all temperatures. This is likely
a mechanism difference (RMG 3.3.0 vs Goldman's 2.4.1 chemistry/thermodynamics).

---
