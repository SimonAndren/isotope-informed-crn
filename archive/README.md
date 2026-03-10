# Archive

This directory contains code, logs, and diagnostic files from the delta-13C
investigation (Mar 2-9, 2026). These files are preserved for reference but are
not part of the active codebase.

See `docs/delta13c_investigation.md` for a summary of the investigation.

## Contents

### experiments/exp1_validate_stage3.py

Experiment script that validated our Stage 3 analysis code path by running
Goldman's DRG mechanism through both `get_delta()` (cluster CSV) and
`get_delta_with_map()` (species map). Both approaches agreed at Table 2 ~ 0.6,
confirming the analysis code was correct and the bug was in Stage 2.

### docs/kie_diagnosis.md

Diagnostic document written during the investigation (Mar 5-8). Lists five
hypotheses for the delta-13C error, their evidence, and recommended experiments.
The root cause turned out to be degeneracy double-counting in `enlarge()`
(not one of the original five hypotheses — it was discovered via a different
line of reasoning).

### docs/vendor-modifications.md

Tracks all changes made to vendored RMG-Py code during the investigation.
Includes the pandas 2.x compatibility fix, try-except KIE guard, simple_fixed
method, and the manual YAML degeneracy patch. Several of these are now
superseded by the `use_original_reactions=True` + `simple_corrected` solution.

### rmg_pipeline/output/*.log

Stage 2 run logs from the various experiments (run-5, run-6, chain runs).

### devlog/

Session journals from each day of the investigation. Not archived (remain in
place) but referenced here for completeness.
