# SQI supplemental protocol experiments

This directory contains the extra protocol audits requested after the
paper-aligned SQI reproduction:

- strict Table 6 feature selection over all 127 non-empty SQI subsets;
- source-level bootstrap and source/provenance-stratified diagnostics;
- fSQI mechanism and flat-threshold sensitivity checks;
- leave-one-SQI-out and cross-noise generalization checks;
- noise-time isolated dataset builder and rerun entrypoints.

Default inputs are the existing paper-aligned artifacts:

```powershell
.\.venv\Scripts\python.exe -m src.supplemental_sqi_experiments.run diagnose-existing
```

Important outputs:

- `outputs/sqi_supplemental/existing_seed0/strict_table6/all_127_subset_val.csv`
- `outputs/sqi_supplemental/existing_seed0/strict_table6/selected_by_cardinality_test.csv`
- `outputs/sqi_supplemental/existing_seed0/model_diagnostics/stratified_score_summary.csv`
- `outputs/sqi_supplemental/existing_seed0/fsqi_mechanism/fsqi_threshold_scan.csv`
- `outputs/reports/sqi_supplemental/existing_seed0/**/fig_supp_*.png`

Noise-time isolated rerun entrypoints:

```powershell
.\.venv\Scripts\python.exe -m src.supplemental_sqi_experiments.run build-isolated --seed 0
.\.venv\Scripts\python.exe -m src.supplemental_sqi_experiments.run run-isolated --seed 0
```

`run-isolated` can be long because it invokes paper QRS detectors and LM-MLP
training.  The generated artifacts stay under `outputs/sqi_supplemental/`.
The default isolated builder uses disjoint NSTDB time regions across
train/validation/test with a 1 s start stride inside each region; this removes
cross-split noise-time leakage while preserving enough 10 s windows for Set-a
balancing.

Multi-seed stability:

```powershell
# SVM on 20 source-grouped resplits
.\.venv\Scripts\python.exe -m src.supplemental_sqi_experiments.run stability

# Full MLP extension: 20 splits x 10 initializations, CPU LM training, long run
.\.venv\Scripts\python.exe -m src.supplemental_sqi_experiments.run stability --include-mlp
```
