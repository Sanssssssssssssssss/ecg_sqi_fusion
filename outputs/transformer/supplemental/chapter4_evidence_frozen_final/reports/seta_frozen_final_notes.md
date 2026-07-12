# Set-A 12-Lead Frozen Final Notes

This is a staging note for Chapter 4 evidence. It does not replace the main report yet.

## Frozen Protocol Audit

- Output root: `outputs\transformer\supplemental\chapter4_evidence_frozen_final`
- Original labeled: `998 = acceptable 773, unacceptable 225`
- Split original: train `541/158`, val `116/34`, test `116/33`.
- Final train: `{'0': 541, '1': 541}`; val/test generated rows `0`.
- SMC candidate counts: `{'original_seta': 998, 'ptb12_morph': 211, 'seta_native_morph': 153, 'noise_style': 19}`.
- SQI feature rerun arms: `{'native_imbalanced': 998, 'fixed_synthetic': 1381, 'quota_draw': 1381, 'smc_gapfill': 1381}`.

## All-Wave Data Repair Metrics

Use this as the AUC??: `c2st_auc` over all waveform-derived construction features, not SQI-domain AUC.

| construction | n_original | n_generated | c2st_auc | rbf_mmd | swd | cdf_gap_mean | pca_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| within_original_resample | 158 | 383 | 0.7070 | 0.0003 | 0.0330 | 0.0233 | 0.8172 |
| fixed_synthetic | 158 | 383 | 0.9038 | 0.5115 | 0.7021 | 0.4394 | 0.5431 |
| quota_draw | 158 | 383 | 0.6446 | 0.0047 | 0.1263 | 0.2303 | 0.9138 |
| smc_gapfill | 158 | 383 | 0.6173 | 0.0036 | 0.1099 | 0.2256 | 0.9086 |

## Repaired Setup Model Comparison

SVM/MLP were rerun from QRS -> 84-SQI -> train-only normalization on this frozen protocol. Conformer is waveform-only.

| model | input | threshold_source | threshold | acc | auc | balanced_acc | acceptable_recall | original_unacceptable_recall | confusion | train_original_bad_recall_fixed05 | val_bad_recall_fixed05 | test_original_bad_recall_fixed05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SQI SVM-RBF selected5 | bSQI,basSQI,kSQI,sSQI,fSQI | validation_max_accuracy | 0.2110 | 0.8792 | 0.8879 | 0.7598 | 0.9741 | 0.5455 | {"tn": 18, "fp": 15, "fn": 3, "tp": 113} |  |  |  |
| SQI SVM-RBF all84 | all84 | validation_max_accuracy | 0.1820 | 0.8658 | 0.9310 | 0.7729 | 0.9397 | 0.6061 | {"tn": 20, "fp": 13, "fn": 7, "tp": 109} |  |  |  |
| SQI LM-MLP 84-4-1 | all84 | validation_max_accuracy | 0.7235 | 0.8993 | 0.8979 | 0.7944 | 0.9828 | 0.6061 | {"tn": 20, "fp": 13, "fn": 2, "tp": 114} |  |  |  |
| 12-lead E31-style waveform comparator | 12-lead waveform-derived channels | validation_recall_balanced | 0.5905 | 0.9195 | 0.8879 | 0.8832 | 0.9483 | 0.8182 | {"tn": 27, "fp": 6, "fn": 6, "tp": 110} | 0.8101 | 0.8235 | 0.7879 |

## Synthesis Values To Keep Blank For Now

Do not copy the old `chapter4_evidence` synthesis/construction-effect values into the manuscript yet. Earlier values could drift because `quota_draw` rebuilt candidate signals with stale generator code. This is now fixed in the transformer experiment, but the paper SQI baseline synthesis protocol should be rerun independently before filling these cells.

Leave blank / mark pending for:

- SQI paper-aligned synthesis split/composition table: original acceptable, original poor, synthetic em, synthetic ma by split.
- SQI paper-aligned SVM/MLP model table on its own synthetic protocol.
- SQI paper-aligned domain-shift panel values: SQI-domain AUC, MMD/permutation p, PCA source-data for original poor vs synthetic em/ma.
- Any table rows named `fixed_synthetic`, `quota_draw`, or old `synthetic em/ma` unless their protocol path and split match the rerun artifacts.

## Commands For The SQI Baseline Thread

Run these from the repository root with `.venv`:

```powershell
.venv\Scripts\python.exe -m src.sqi_pipeline.cli --profile paper_aligned --artifacts_dir outputs/sqi_paper_aligned_ch4_rerun --fresh --force --seed 0
.venv\Scripts\python.exe -m supplemental_sqi_experiments.run --artifacts-dir outputs/sqi_paper_aligned_ch4_rerun --out-dir outputs/sqi_supplemental/ch4_rerun --report-dir reports/sqi_supplemental/ch4_rerun strict-table6 --force
.venv\Scripts\python.exe -m supplemental_sqi_experiments.run --artifacts-dir outputs/sqi_paper_aligned_ch4_rerun --out-dir outputs/sqi_supplemental/ch4_rerun --report-dir reports/sqi_supplemental/ch4_rerun final-claims --force
.venv\Scripts\python.exe -m supplemental_sqi_experiments.run --artifacts-dir outputs/sqi_paper_aligned_ch4_rerun --out-dir outputs/sqi_supplemental/ch4_rerun --report-dir reports/sqi_supplemental/ch4_rerun model-diagnostics --include-mlp --force
```

Ask that thread to return:

- `outputs/sqi_paper_aligned_ch4_rerun/config/run_summary_seed0.json`
- `outputs/sqi_paper_aligned_ch4_rerun/splits/split_seta_seed0_paper_balanced.csv` and its audit CSV
- `outputs/sqi_paper_aligned_ch4_rerun/models/svm/table5_12lead_single_sqi_seed0.csv`
- `outputs/sqi_paper_aligned_ch4_rerun/models/svm/table6_12lead_combo_sqi_seed0.csv`
- `outputs/sqi_paper_aligned_ch4_rerun/models/svm/table7_svm_selected5_seed0.csv`
- `outputs/sqi_paper_aligned_ch4_rerun/models/lm_mlp/lm_mlp_test_metrics_seed0.json`
- final-claims source-data/CSV files for domain shift, especially original poor vs synthetic em/ma AUC/MMD/PCA values

## Transformer Frozen Reproduction Commands

```powershell
.venv\Scripts\python.exe -m supplemental_transformer_experiments.chapter4_evidence.run --out chapter4_evidence_frozen_final --force seta-build --run
.venv\Scripts\python.exe -m supplemental_transformer_experiments.chapter4_evidence.run --out chapter4_evidence_frozen_final --force seta-sqi --run
.venv\Scripts\python.exe -m supplemental_transformer_experiments.chapter4_evidence.run --out chapter4_evidence_frozen_final --force seta-repair --run
.venv\Scripts\python.exe -m supplemental_transformer_experiments.chapter4_evidence.run --out chapter4_evidence_frozen_final --force seta-models --run
```
