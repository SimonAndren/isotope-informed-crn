# KIE Error Diagnosis — Why Our Mechanism Gives Wrong δ¹³C

**Status as of 2026-03-08.** Table 2 score: Ours = 7.9–8.7 (target ≤ 0.6, Goldman full = 0.5)

---

## Background

Goldman (2019) used RMG 2.4.1 to generate a propane pyrolysis mechanism, expanded it to include
all ¹³C isotopologues, and simulated with Cantera to reproduce Gilbert et al. (2016) experimental
δ¹³C data. His Table 2 metric (std of scaled deviations from experiment) = 0.5 for the full model.

We ported this to RMG 3.3.0. Our mechanism scores 7.9–8.7 — roughly 15× worse than Goldman's.
δCH4 at 850°C is +150‰ to +462‰ instead of Goldman's −21‰.

---

## Confirmed Facts

| Fact | Evidence |
|------|----------|
| Goldman's full_model YAML gives Table 2 = 0.5 via our compare script | `compare_to_goldman.py` uses `compute_slopes_goldman()` |
| Both our yamls give ~60% propane conversion at 850°C | Cantera test, Mar 6 |
| Carbon mass balance is conserved | ¹²C-only propane → zero ¹³CH4 |
| C(31) = ¹²CH4, C(32) = ¹³CH4 | species_dictionary.txt confirmed |
| He, CCC(2–6), propane_ooo(1) all present in mechanism | diagnostic check, Mar 6 |
| Stage 1 mechanisms are byte-for-byte identical (old vs new) | diff comparison |
| No labeled reactions are pruned during Stage 2 | Code review of isotopes.py `run()` flow |
| Our base mechanism is a SUPERSET of Goldman's DRG model | Species comparison, Mar 8 |

---

## Run Summary

| Run | KIE Method | Species/Rxns | δCH4@850°C | Table 2 |
|-----|------------|--------------|------------|---------|
| run-1 | `simple` (Goldman original) | — | CRASH | — |
| run-2 | `simple_fixed` (our patch) | 187 / 5086 | +150‰ | 7.9 |
| run-5 | `none` v3 (tree-estimated) | 187 / 5086 | +430‰ | 8.5 |
| run-6 | `none` v4 (deepcopy, no KIE) | 187 / 5086 | +462‰ | 8.7 |
| Goldman full | `simple` (RMG 2.4.1) | 343 / 7096 | −21‰ | 0.5 |
| Goldman DRG | `simple` (RMG 2.4.1) | 31 / 167 | ~−21‰ | 0.6 |

---

## Goldman's Models — Key Comparison

| Model | Base spp | Total spp | Rxns | Table 2 |
|-------|----------|-----------|------|---------|
| 3-rxn | 10 | 24 | 19 | 1.7 |
| 6-rxn | 12 | 26 | 35 | 6.5 |
| DRG | **10** | 31 | 167 | **0.6** |
| Full | 31 | 343 | 7096 | 0.5 |
| **Ours** | **31** | **187** | **5086** | **7.9** |

Goldman's DRG model has only 10 base species — ALL of which are in our mechanism (just
different naming: CCC→propane_ooo(1), [He]→He, C=C→C2H4(33), etc.). Our mechanism has those
same 10 plus 21 additional species (allyl, propene, vinyl, acetylene, butene, etc.).

**The problem is NOT missing species.** A 10-species mechanism with correct kinetics scores 0.6.
Our 31-species mechanism scores 7.9. The problem must be in the kinetics of labeled reactions.

---

## The Central Mystery

Run-6 (KIE `none` v4): deepcopy of base kinetics → all isotopologues get identical rate
constants. With no isotope discrimination, δ¹³C should be conserved: δCH4 ≈ −28‰.
Actual: **+462‰**. This is physically impossible.

The 0.18% KIE factor in `simple_fixed` cannot explain the 312‰ gap between run-2 (+150‰)
and run-6 (+462‰). Something structural is different about how reactions are processed.

---

## Hypotheses

### H1: Our Stage 3 analysis code has a bug (δ calculation or initial conditions)

**Mechanism:** `compute_slopes_ours()` uses `OUR_SPECIES_MAP` and `our_init_x()` for species
mapping and initial ¹³C distribution. Goldman's compare path uses `compute_slopes_goldman()`
with `goldman_init_x()` and his cluster CSV. If our mapping is wrong (e.g., CCC(5) assigned
wrong position, OUR_CLUSTERS has wrong cluster numbers, or `our_init_x` distributes ¹³C
incorrectly), the δ values would be systematically wrong.

**Why it's plausible:** Goldman's full_model gives Table 2 = 0.5 through our script, but that
uses a SEPARATE code path (`compute_slopes_goldman` + `goldman_init_x`). The code path for
our mechanism (`compute_slopes_ours` + `our_init_x` + `OUR_SPECIES_MAP`) has never been
validated against a known-good mechanism.

**Key test:** Take Goldman's DRG YAML, map species names to our convention, and run through
`compute_slopes_ours()`. If it still gives ~0.6, our analysis code is correct. If not, the
bug is in OUR_SPECIES_MAP, our_init_x, or get_delta_ours.

### H2: `cluster[-1]` is not always the unlabeled base reaction

**Mechanism:** All three KIE methods assume `cluster[-1]` is the unlabeled (base) reaction.
The `cluster()` function builds clusters by popping from the end of `core.reactions`. If
the unlabeled reaction is not the last to be popped into a cluster, `cluster[-1]` could be
a labeled reaction. Deepcopy would then copy wrong kinetics (tree-estimated, potentially
with very different A/b/Ea) from a labeled reaction to all other labeled reactions.

**Why it's plausible:** No explicit test has verified this assumption. Goldman's code has the
same assumption (`# set unlabeled reaction as the standard to compare`) but in RMG 2.4.1
all reactions had correct kinetics anyway, so a wrong base would be harmless. In RMG 3.3.0
with tree-estimated kinetics, a wrong base would corrupt the entire cluster.

**Why it could explain +462‰:** If even 10-20% of clusters have a labeled reaction as
`cluster[-1]`, deepcopy would spread tree-estimated kinetics to those clusters, causing
massive isotope fractionation.

### H3: The `simple_fixed` per-reaction guard causes structurally different behavior from `none` v4

**Mechanism:** In `simple_fixed` (line 670), `get_labeled_reactants(reaction, family)` is
called for EACH labeled reaction inside the try block. If it raises for any reaction, the
ENTIRE cluster is skipped — but reactions already deepcopied earlier in the loop retain
their new kinetics (without `change_rate`). In `none` v4, only the base is checked; all
labeled reactions get deepcopied if the base passes.

This means `simple_fixed` and `none` v4 produce different sets of corrected reactions:
- `simple_fixed`: some clusters partially corrected (first N reactions deepcopied, rest
  keep tree-estimated), some fully skipped
- `none` v4: clusters are all-or-nothing (all corrected or all skipped)

**Why it's plausible:** The 312‰ gap between `simple_fixed` (+150‰) and `none` v4 (+462‰)
is far too large for a 0.18% A-factor difference. A structural difference in which reactions
get corrected would explain this. The partial-correction behavior in `simple_fixed` could
accidentally produce better routing than `none` v4's all-or-nothing approach.

### H4: RMG 3.3.0 generates different base reactions than Goldman's RMG 2.4.1

**Mechanism:** Our Stage 1 produces 31 species / 150 reactions. Goldman's base had 31
species / 191 reactions. Same species count but 41 fewer reactions. These missing reactions
are reaction pathways that RMG 3.3.0's database doesn't generate (or generates with different
kinetics). During Stage 2, the isotopologue expansion inherits these base reactions. If key
propane cracking pathways have different rate constants in our base vs Goldman's, the ¹³C
routing through the network would differ.

**Why it's plausible:** The RMG kinetics database changed substantially between 2.4.1 and
3.3.0 (new training data, different tree structures). Even for the same reaction family, the
rate rules and tree estimates can be very different.

**Why it's NOT about mechanism size:** Goldman's DRG model (10 base species, 167 reactions)
scores 0.6. Our mechanism (31 base species, 5086 reactions) scores 7.9. More species/reactions
doesn't help if the kinetics are wrong.

### H5: Singleton or mis-clustered labeled reactions retain uncorrected tree-estimated kinetics

**Mechanism:** If `compare_isotopomers()` fails to match some labeled reactions to their
unlabeled counterparts, those reactions form singleton clusters. The KIE loop iterates over
`cluster[:-1]` which is empty for singletons — so these reactions are never corrected by
any KIE method. They retain tree-estimated kinetics (potentially orders of magnitude wrong
in A, b, or Ea).

Similarly, if a labeled reaction is misclustered (grouped with the wrong base reaction),
deepcopy would give it kinetics from an unrelated reaction.

**Why it's plausible:** The `compare_isotopomers` function uses `remove_isotope` + structural
comparison. If isotope removal changes the resonance structures or atom ordering, matching
could fail for some species.

---

## Ruled Out

| Hypothesis | Why ruled out |
|------------|---------------|
| Missing C5/C6 species | DRG model has 10 base species (no C5/C6) and scores 0.6. Network size is not the issue. |
| Propane→methane mapping error | OUR_SPECIES_MAP explicitly separates clusters 26 (propane) and 19 (methane). |
| Wrong n_13C counts | Carbon conservation passes. |
| 2-atom vs 3-atom KIE | KIE magnitude is ≤1% regardless. Cannot explain 100+ ‰ errors. |
| Species naming (He vs [He]) | Confirmed "He" is correct in mechanism. |
| Mass balance leak | ¹²C-only propane → zero ¹³CH4. |
| Labeled reaction pruning in Stage 2 | Code review: no filtering/pruning occurs. All reactions are kept. |

---

## Experiments (Priority Order)

### Exp 1: Validate Stage 3 with Goldman's DRG mechanism (tests H1) ⭐ DO FIRST

Take Goldman's DRG YAML (10 base species, 31 total, 167 reactions, known Table 2 ≈ 0.6).
Create a species name mapping (CCC → propane_ooo(1), [He] → He, etc.) and run through
`compute_slopes_ours()` with `our_init_x()`.

- If result ≈ 0.6 → our analysis pipeline is correct. Problem is in Stage 2 mechanism.
- If result ≈ 7+ → bug in OUR_SPECIES_MAP, our_init_x, or get_delta_ours.

**Cost:** ~5 min (no Stage 2 needed, just Cantera simulation with name mapping).
**Impact:** If our analysis code is wrong, EVERYTHING else is moot. Must validate first.

### Exp 2: Compare reaction rate constants between yamls (tests H3, H5)

Load both yamls in Cantera and compare every reaction's forward rate constant at 850°C.
Identify reactions where the rate differs by more than 1%:

```python
gas_sf = ct.Solution("output_backup/isotope/iso/chem.yaml")   # simple_fixed
gas_none = ct.Solution("output/isotope/iso/chem.yaml")         # none v4
gas_sf.TP = 1123, 2e5
gas_none.TP = 1123, 2e5
for i in range(gas_sf.n_reactions):
    ratio = gas_sf.forward_rate_constants[i] / gas_none.forward_rate_constants[i]
    if abs(ratio - 1.0) > 0.01:
        print(f"Rxn {i}: {gas_sf.reaction(i).equation}  ratio={ratio:.4f}")
```

**Cost:** ~1 min (no re-run needed).
**Impact:** Directly shows which reactions differ between `simple_fixed` and `none` v4,
explaining the 312‰ gap.

### Exp 3: Verify `cluster[-1]` is always unlabeled (tests H2)

Add assertion to `apply_kinetic_isotope_effect_simple_fixed` before deepcopy:

```python
base_reaction = cluster[-1]
for sp in base_reaction.reactants + base_reaction.products:
    for atom in sp.molecule[0].atoms:
        assert atom.element.isotope == -1, \
            f"cluster[-1] is LABELED in cluster {index}: {sp.label}"
```

Run Stage 2 with this assertion. If it fires, H2 is confirmed.

**Cost:** ~17 min (one Stage 2 run).
**Impact:** If confirmed, explains the entire problem. Fix is trivial (find the
unlabeled reaction explicitly instead of assuming it's last).

### Exp 4: Count singleton and partially-corrected clusters (tests H3, H5)

Add detailed logging to both `simple_fixed` and `none` v4:
- Total clusters processed
- Clusters in supported families
- Clusters where base guard passed
- Clusters where per-reaction guard failed mid-loop (simple_fixed only)
- Singleton clusters (len(cluster) == 1)
- Total reactions corrected vs uncorrected

**Cost:** ~34 min (two Stage 2 runs).
**Impact:** If many clusters are singletons or partially corrected, explains why
deepcopy doesn't fully fix the kinetics.

### Exp 5: Compare our base reaction kinetics to Goldman's (tests H4)

For the ~10 key reactions shared between our mechanism and Goldman's DRG model
(H-abstraction from propane, CH3 recombination, β-scission, etc.), compare
rate constants at 800–950°C. If our base kinetics differ significantly from
Goldman's, the Stage 2 isotopologue expansion inherits wrong rate constants
regardless of KIE correction.

**Cost:** ~30 min (manual comparison of specific reactions in both yamls).
**Impact:** If base kinetics are wrong, we need a different RMG database or
Goldman's exact kinetics.

---

## Recommended Experiment Order

```
Exp 1 (validate analysis code)     ~5 min    ← must do first
  ↓
Exp 2 (compare yaml rate constants) ~1 min   ← quick, high info
  ↓
Exp 3 (cluster[-1] assertion)       ~17 min  ← tests most dangerous hypothesis
  ↓
Exp 4 (cluster statistics)          ~34 min  ← if Exp 3 passes
  ↓
Exp 5 (base kinetics comparison)    ~30 min  ← if all above pass
```

If Exp 1 reveals a bug in our analysis code, fix it before running anything else.
If Exp 3 confirms wrong cluster[-1], fix the cluster ordering and re-run Stage 2.
