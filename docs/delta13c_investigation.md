# Delta 13C Discrepancy Investigation — Summary

**Dates:** 2026-03-02 through 2026-03-09
**Outcome:** Root cause identified and fixed. Table 2 score improved from 7.9 to 1.2
(Goldman reference: 0.5).

---

## Problem Statement

After replicating Goldman et al. (2019)'s propane pyrolysis isotopologue pipeline
using RMG 3.3.0, our generated mechanism produced wildly incorrect delta-13C values:

- **Our delta-CH4 at 850C:** +150 to +462 permil
- **Goldman's delta-CH4 at 850C:** -28 permil
- **Our Table 2 score:** 7.9-8.7 (std of scaled deviations from Gilbert 2016 experiment)
- **Goldman's Table 2 score:** 0.5

The sign and magnitude of the error indicated a fundamental problem in how 13C was
routed through the reaction network, not a minor kinetics issue.

---

## Investigation Timeline

### Phase 1: Baseline Reproduction (Mar 2-3)

- Confirmed Goldman's pre-generated YAML gives Table 2 = 0.5 through our analysis code
- Ran full RMG pipeline from scratch: 31 base species / 150 reactions ->
  187 isotopologue species / 5086 reactions (Goldman: 343 / 7096)
- First delta-CH4 reading: +150 permil (run-2, simple_fixed KIE)

### Phase 2: KIE Method Investigation (Mar 3-6)

Tested multiple KIE application methods to isolate the error source:

| Run | KIE Method | delta-CH4 at 850C | Table 2 | Key Finding |
|-----|------------|-------------------|---------|-------------|
| run-1 | `simple` (Goldman original) | CRASH | - | Crashes on R_Addition_MultipleBond families |
| run-2 | `simple_fixed` (deepcopy base kinetics) | +150 | 7.9 | Tree-estimated kinetics partially corrected |
| run-5 | `none` v3 (tree-estimated, no correction) | +430 | 8.5 | Raw tree estimates are 3-4x wrong |
| run-6 | `none` v4 (deepcopy base, no KIE factor) | +462 | 8.7 | **Pure routing test: physically impossible result** |

**Critical insight from run-6:** With identical kinetics for all isotopologues (no
isotope discrimination whatsoever), delta-CH4 should equal the initial propane value
(~-28 permil). Getting +462 permil is physically impossible — it requires creating
13C from nothing. This proved the bug was in the mechanism structure itself, not in
the KIE factors.

### Phase 3: Root Cause Discovery (Mar 9)

Analyzed the reaction degeneracies in the expanded mechanism and found the primary bug.

---

## Root Cause: Degeneracy Double-Counting in `enlarge()`

When `use_original_reactions=False`, RMG's `enlarge()` function re-enumerates all
reactions among isotopologue species from scratch. The graph isomorphism algorithm
treats 12C and 13C as equivalent when computing reaction degeneracies, which causes
double-counting when isotope substitution breaks molecular symmetry.

**Example — propane with one 13C at an edge position (CCC(4)):**

The molecule has two methyl groups: 13CH3 (3 equivalent H's) and 12CH3 (3 equivalent
H's). For H-abstraction:

- **Correct treatment:** Two distinct reaction channels, each with degeneracy=3.
  Total rate = 2 x k_template x 3 = 6 x k_template (same as unlabeled propane).

- **What `enlarge()` does:** Graph isomorphism sees all 6 primary H atoms as
  equivalent (ignoring the 13C/12C distinction). Result: two reactions each with
  degeneracy=6. Total rate = 2 x k_template x 6 = 12 x k_template — **2x too fast**.

**Consequences:**
- Labeled propane isotopologues crack faster than unlabeled propane
- 13C-containing fragments are over-produced relative to 12C fragments
- delta values drift massively positive (13C appears to accumulate in products)
- Propane conversion was 60% (vs Goldman's 42.6%) because reactions were overcounted

**Evidence:**
- Cluster 105 analysis: 54 reactions with deg=6 in our mechanism vs 31 in Goldman's
  base — 23 extra overcounted channels
- Switching to `use_original_reactions=True` reduced reactions from 5086 to 4750
  (336 overcounted channels removed)

### Secondary Issue: Tree-Estimated Kinetics (RMG 3.3.0)

RMG 3.3.0 does not match labeled reactions to training data. Instead, it falls back
to tree-node estimates with completely different Arrhenius parameters:

- Base (12CH3 + 12CH3, training match): A=9.45e14, b=-0.538, Ea=0.135
- Labeled (13CH3 + 13CH3, tree estimate): A=9.59e10, b=+0.611, Ea=0.0
- **Rate error at 850C: 3-4x**

This was masked by the larger degeneracy bug but contributed to the initial +150 permil
reading under `simple_fixed`.

### Tertiary Issue: `cluster[-1]` Assumption

The KIE code assumed `cluster[-1]` was always the unlabeled base reaction. In 9.2% of
clusters (14 out of 152), a labeled reaction was at `cluster[-1]`. When `simple_fixed`
deepcopied "base" kinetics from `cluster[-1]`, it sometimes copied tree-estimated
kinetics from a labeled reaction to all other reactions in the cluster.

---

## Solution

### 1. Switch to `use_original_reactions=True`

Instead of re-enumerating reactions from scratch via `enlarge()`, this path takes
base reactions from Stage 1 and generates isotopologue variants using
`generate_isotope_reactions()`. Degeneracy is scaled correctly:

    A_labeled = A_base x (new_deg / old_deg)

This handles symmetry-breaking from isotope substitution properly because the
degeneracy ratio accounts for exactly how many equivalent atom permutations exist
in the labeled species versus the base.

### 2. New `simple_corrected` KIE Method

Previous KIE methods (`simple_fixed`, `none`) deepcopy base kinetics, which
**overwrites** the degeneracy correction applied by `generate_isotope_reactions`.
The new `simple_corrected` method applies KIE directly on the existing (already
degeneracy-corrected) kinetics:

    reaction.kinetics.change_rate(sqrt(mu_base / mu_labeled))

This preserves the degeneracy scaling while adding the reduced-mass isotope effect.
Final kinetics: A_base x (new_deg/old_deg) x sqrt(mu_base/mu_labeled).

### 3. Helper `_find_unlabeled_reaction()`

Replaces the `cluster[-1]` assumption with an explicit search through the cluster
for the first reaction with no isotope labels. Fixes the 9.2% of clusters where
a labeled reaction was incorrectly used as the base.

---

## Results

| Run | Method | delta-CH4 at 850C | Table 2 |
|-----|--------|-------------------|---------|
| run-6 | `enlarge()` + no KIE | +462 | 8.7 |
| run-2 | `enlarge()` + simple_fixed | +150 | 7.9 |
| **run-8** | **use_original_reactions + disabled KIE** | **-30.5** | **1.2** |
| **run-9** | **use_original_reactions + simple_corrected** | **-28.4** | **1.2** |
| Goldman full | RMG 2.4.1 | -28.3 | 0.5 |

**Slope comparison (run-9 vs Goldman full model):**

```
                  800    850    900    950
dC2H4 = f(dCH4)  0.516  0.546  0.586  0.606   (Goldman: 0.502  0.514  0.545  0.576)
dC2H6 = f(dCH4)  0.979  0.978  0.975  0.977   (Goldman: 0.979  0.977  0.975  0.985)
dBulk = f(dCH4)  0.665  0.666  0.671  0.673   (Goldman: 0.666  0.667  0.672  0.680)
```

The dC2H6 and dBulk slopes now match Goldman to within 0.01. The remaining gap is
in dC2H4, which is systematically ~0.02-0.03 higher than Goldman at all temperatures.

---

## Remaining Gap: Table 2 = 1.2 vs Goldman's 0.5

The remaining gap is likely due to:

1. **Mechanism chemistry differences** between RMG 3.3.0 and Goldman's 2.4.1 — the
   kinetics database has evolved, giving different rate constants for the same
   reaction families. The dC2H4 slope deviation is systematic and temperature-
   dependent, consistent with a kinetics difference rather than a structural bug.

2. **Missing C5/C6 species** — our mechanism has 187 isotopologue species vs
   Goldman's 343. The additional species in Goldman's model provide extra
   reaction channels that affect 13C routing. However, Goldman's DRG model
   (31 species, Table 2 = 0.6) shows this is a minor contribution.

3. **Entropy-based isotope effect** — `correct_entropy()` in RMG's isotopes.py
   applies symmetry-based entropy corrections to isotopologues. This creates a
   thermodynamic isotope effect (~2.5 permil) even with zero KIE. This is
   physically correct but may be handled differently in Goldman's RMG 2.4.1.

---

## Open Questions for Future Work

1. **Automated species annotation** — The current species mapping
   (`OUR_SPECIES_MAP`) is manually constructed by inspecting
   `species_dictionary.txt` after each Stage 2 run. Species indices change when
   Stage 1 runs inline vs loading from saved CHEMKIN. This mapping needs to be
   automated.

2. **Entropy-only isotope effect** — When KIE is disabled (run-8), there is still
   a ~2.5 permil offset from Goldman. This comes from the symmetry-number entropy
   correction in `correct_entropy()`. Need to understand whether Goldman's mechanism
   includes this correction and how it interacts with the KIE.

3. **QIRN vectorized approach** — The QIRN code (Mueller & Wu, 2022) carries isotope
   fractions as a vector alongside the base mechanism ODE. This avoids the
   combinatorial explosion of isotopologue species entirely. The next phase is to
   implement QIRN-style tracking and validate that it reproduces the Goldman results
   with the same fidelity as our RMG-expanded approach.
