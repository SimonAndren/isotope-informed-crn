# Vendor Code Modifications & Pipeline Assumptions

This file tracks every change made to vendored third-party code and every assumption or
deviation from Goldman 2019. Update it whenever a new modification is made.

---

## vendor/RMG-Py/rmgpy/tools/isotopes.py

### Mod 1 — pandas 2.x compatibility (`DataFrame.append` removal)

**Line:** ~757 (inside `store_flux_info`, nested in `ensure_correct_degeneracies`)

**Problem:** `DataFrame.append()` was removed in pandas 2.0. The rmg_env uses pandas 2.3.1,
so Stage 2 crashed during degeneracy checking after the expensive reaction enumeration had
already completed.

**Change:**
```python
# BEFORE
return product_list.append({'product': species, 'flux': flux,
                            'product_struc_index': structure_index,
                            'symmetry_ratio': symmetry_ratio},
                           ignore_index=True)

# AFTER
new_row = pd.DataFrame([{'product': species, 'flux': flux,
                          'product_struc_index': structure_index,
                          'symmetry_ratio': symmetry_ratio}])
return pd.concat([product_list, new_row], ignore_index=True)
```

**Risk:** Low. Semantically identical; pandas docs confirm `pd.concat` is the correct
replacement.

**Should have:** Asked user before modifying vendor code. (Violates "never assume" rule.)

---

### Mod 2 — KIE try-except guard in `apply_kinetic_isotope_effect_simple`

**Lines:** ~589–597

**Problem:** Stage 2 crashed every time just before `save_everything()`, after ~16 minutes
of successful reaction enumeration. The crash occurred inside
`apply_kinetic_isotope_effect_simple` — either `family.add_atom_labels_for_reaction()` or
`get_reduced_mass()` raised an exception for certain reaction clusters. The traceback went
to stderr (not captured), so the exact error was not confirmed.

**Change:** Wrapped the per-cluster KIE application in a try-except that logs a warning
and skips the cluster on failure.

```python
# BEFORE
logging.debug('modifying reaction rate for cluster {0}...'.format(index, family.name))
reaction = cluster[-1]
labeled_reactants = get_labeled_reactants(reaction, family)
base_reduced_mass = get_reduced_mass(labeled_reactants, labels, three_member_ts)
for reaction in cluster[:-1]:
    labeled_reactants = get_labeled_reactants(reaction, family)
    reduced_mass = get_reduced_mass(labeled_reactants, labels, three_member_ts)
    reaction.kinetics.change_rate(math.sqrt(base_reduced_mass / reduced_mass))

# AFTER
logging.debug('modifying reaction rate for cluster {0}...'.format(index, family.name))
try:
    reaction = cluster[-1]
    labeled_reactants = get_labeled_reactants(reaction, family)
    base_reduced_mass = get_reduced_mass(labeled_reactants, labels, three_member_ts)
    for reaction in cluster[:-1]:
        labeled_reactants = get_labeled_reactants(reaction, family)
        reduced_mass = get_reduced_mass(labeled_reactants, labels, three_member_ts)
        reaction.kinetics.change_rate(math.sqrt(base_reduced_mass / reduced_mass))
except Exception as e:
    logging.warning('isotope: KIE application failed for cluster {0} (family {1}): {2}. '
                    'Skipping KIE for this cluster.'.format(index, cluster[0].family, e))
```

**Risk:** UNKNOWN. If many clusters are skipped, the KIE correction will be incomplete and
reaction rates will deviate from Goldman's values. The actual error was never confirmed —
the fix was applied before root-cause diagnosis was complete.

**Should have:** Captured the actual traceback first (redirect stderr), identified the
exact exception, then presented options to the user before patching.

**TODO:** After Stage 2 completes, check the log for any "KIE application failed" warnings
to understand how many clusters were affected and whether this matters.

---

### Mod 3 — `apply_kinetic_isotope_effect_simple_fixed`

**Lines:** Added after line ~602 (after `apply_kinetic_isotope_effect_simple`, before
`get_labeled_reactants`).

**Problem:** `apply_kinetic_isotope_effect_simple` calls `reaction.kinetics.change_rate(factor)`
on whatever kinetics the labeled reaction has. If that kinetics was tree-estimated (wrong b/Ea),
the result is a factored-up wrong rate — not a small KIE correction on the correct base rate.

**Change:** New function (original untouched) with one added line before `change_rate`:
```python
# Copy base reaction kinetics before applying KIE factor.
# This ensures labeled variants inherit the correct b and Ea, not a tree estimate.
reaction.kinetics = _copy.deepcopy(base_reaction.kinetics)
reaction.kinetics.change_rate(math.sqrt(base_reduced_mass / reduced_mass))
```

**Dispatched via:** `kinetic_isotope_effect='simple_fixed'` string in `run()`.

**Risk:** Low. Deepcopy is safe. The only effect is that labeled reactions now get
`b` and `Ea` from the base (unlabeled) reaction, which is what Goldman's model shows.

---

## rmg_pipeline/run_pipeline.py

### Assumption 1 — `use_original_reactions=False`

Passed to `run_isotopes(use_original_reactions=False)`. This means Stage 2 does a **full
re-enumeration** of all reactions among the isotopologue species rather than re-using
reactions from Stage 1. This matches Goldman's approach but is the most expensive option.

### Assumption 2 — `maximum_isotopic_atoms=1000000`

Effectively no limit. Goldman used a large value to allow unlimited ¹³C labeling.

### Assumption 3 — `kinetic_isotope_effect="simple_fixed"` (updated from `"simple"`)

Originally used `"simple"` (Goldman's method). Changed to `"simple_fixed"` after discovering
the root cause of the δCH4 = +265‰ bug (should be ~−23‰).

**Root cause:** RMG 3.3.0 fails to match labeled reactions to training data → falls back to
tree-node estimates with completely different b and Ea (e.g., R25: b=+0.611, Ea=0.0 vs
correct b=−0.538, Ea=0.135). The original `apply_kinetic_isotope_effect_simple` only
multiplied the A factor of the (wrong) tree estimate and left wrong b/Ea unchanged.

**Fix:** New function `apply_kinetic_isotope_effect_simple_fixed` in `isotopes.py` (see
Mod 3 below). The original `"simple"` path is unchanged for comparison.

**Impact:** Expected to fix the false 3-4× KIE at 850°C and bring δCH4 from +265‰ to
near −23‰ (Goldman's value). Re-run Stage 2 to validate.

### Assumption 4 — Stage 1 `toleranceMoveToCore=0.1` produces 31 species, not Goldman's 31+C5/C6

Our Stage 1 generates 31 core species (matching Goldman's count), but the C5/C6 species
that appear in Goldman's full model are not promoted to core in our run. They exist in the
edge (160 edge species) but are discarded. This is because the RMG 3.3.0 database has
updated (lower) rate constants for alkyne-coupling pathways vs Goldman's 2019 database.

**Impact:** Our Stage 2 produces ~187 isotopologue species vs Goldman's 343. We are missing
the C5/C6 cluster. This was accepted without explicitly asking the user.

**TODO:** Decide whether to lower `toleranceMoveToCore` to include C5/C6 species, or
accept the smaller model for now.

### Assumption 5 — YAML degeneracy fix for mixed CH3+CH3 recombination

After Stage 2 with `simple_fixed` KIE, `rmg_pipeline/output/isotope/iso/chem.yaml` was
patched directly: the mixed recombination `[CH3](14) + [CH3](15) <=> CC(9)` had its A
factor multiplied by 2 (from 9.267885e+14 to 1.853577e+15).

**Root cause:** For R_Recombination with two IDENTICAL base-species radicals, RMG's base rate
(¹²CH3+¹²CH3) accounts for homospecific reactants (degeneracy=1). When the reactants become
distinguishable (¹²CH3 ≠ ¹³CH3), the reaction path degeneracy doubles. Goldman's mechanism
correctly shows A=1.854e+15 ≈ 2×A_base × KIE_factor for the mixed CH3+CH3 reaction.

Our `apply_kinetic_isotope_effect_simple_fixed` deepcopies the base kinetics without adjusting
for the increased degeneracy. Only 1 such R_Recombination exists in our mechanism.

**Effect:** δCH4 at 850°C improved from +448‰ → +153‰ (was −23‰ in Goldman). Table 2 score
improved from 8.7 → 7.9.

**Remaining gap:** Still ~170‰ too positive at 850°C, likely due to missing C5/C6 species
(our 187-species model vs Goldman's 343-species model) and possibly tree-estimated
Disproportionation rates for labeled C3H7+C3H7 pairs.

**NOTE:** This YAML patch is manual and will be lost if Stage 2 is re-run. The correct long-term
fix is to update `apply_kinetic_isotope_effect_simple_fixed` to detect heterospecific
same-base reactors and apply the 2× degeneracy factor.

---

## Crash Recovery Logic

`run_pipeline.py` detects a partial Stage 2 run (`iso/` directory exists but
`iso/chemkin/chem_annotated.inp` is absent) and removes `iso/` before restarting.

The `--yes` / `-y` flag was added to skip the interactive confirmation prompt (needed for
backgrounded runs where `input()` raises `EOFError`).

**Risk:** Auto-removal of partial output on `--yes` is irreversible. Only use `--yes` for
background runs that are known to be restarting from a known-clean state.

The `--force-expand` flag removes a **completed** iso/ directory to allow re-running Stage 2
(e.g., after changing the KIE method). Use with `--yes` for unattended runs:

```bash
conda run --prefix /opt/homebrew/Caskroom/miniforge/base/envs/rmg_env \
    python rmg_pipeline/run_pipeline.py --stage expand --force-expand --yes
```
