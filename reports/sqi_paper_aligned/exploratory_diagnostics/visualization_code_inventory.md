# SQI Visualization Code Inventory

This inventory was built from `src/sqi_pipeline` by searching for plotting imports, `savefig`, and `.png` / `.pdf` outputs. It focuses on the traditional SQI reproduction line, not the transformer/external-benchmark line.

## Summary

The SQI codebase contains more visualization paths than the final report figures and the exploratory diagnostics contact sheet. The current paper-aligned run has these image families actually present:

- Final report figures in `reports/sqi_paper_aligned/figures/`.
- Exploratory diagnostics in `reports/sqi_paper_aligned/exploratory_diagnostics/`.
- Paper-aligned model/QC plots in `outputs/sqi_paper_aligned/`.

The older baseline `outputs/sqi/` directory is not present in the current workspace, so older baseline-profile QC figures referenced by several scripts are code paths, not currently available image files.

## Code Inventory

| Source | Intended outputs | What it visualizes | Current status | Covered by exploratory diagnostics? |
|---|---|---|---|---|
| `src/sqi_pipeline/data/make_split_seta.py` | `split_seta_seed{seed}_label_counts.png` | Original Set-a split label counts. | Old `outputs/sqi/` output missing. | Covered conceptually by `diag_01_split_balance`. |
| `src/sqi_pipeline/noise/make_paper_aligned_balanced_cases.py` | `.label_counts.png`, configured as `paper_balanced_seta_seed0_label_counts.png` | Paper-aligned balanced Set-a class counts after synthetic poor cases. | Present in `outputs/sqi_paper_aligned/qc/`. | Covered and expanded by `diag_01_split_balance`. |
| `src/sqi_pipeline/diagnostics/raw_qc_seta.py` | `{record_id}_wave12.png`, `{record_id}_std_range.png`, `{record_id}_psd.png` | Raw 500 Hz Set-a examples: 12-lead waveform, per-lead std/range, mean PSD. | Old `outputs/sqi/` output missing. | Partially covered by `diag_08_resampling_wave_psd` and `diag_09_qrs_overlay_12lead`; raw std/range is not covered. |
| `src/sqi_pipeline/diagnostics/qc_resample_125.py` | `{rid}_wave_compare_2s.png`, `{rid}_psd_compare.png` | 500 Hz versus 125 Hz waveform/PSD sanity check. | Old `outputs/sqi/` output missing. | Covered by `diag_08_resampling_wave_psd`. |
| `src/sqi_pipeline/diagnostics/qc_resample_alias_check.py` | `{rid}_psd_overlay.png`, `{rid}_time_overlay.png`, `{rid}_psd_diff.png`, `{rid}_psd_relerr.png`, `{rid}_time_diff_2s.png`, `{rid}_time_low40_diff_2s.png` | Current decimation versus steep reference low-pass/downsample aliasing check. | Old `outputs/sqi/` output missing. | Not fully covered; `diag_08` is simpler and does not include the reference-filter error plots. |
| `src/sqi_pipeline/diagnostics/qc_qrs_cache.py` | `{rid}_rpeaks_12lead.png`, `beats_count_by_lead_{det}.png`, `hr_by_lead_{det}.png` | Detector overlay examples, beat-count boxplots, HR boxplots. | Old `outputs/sqi/` output missing; script labels old detectors as `xqrs/gqrs`. | Covered for paper detectors by `diag_02_qrs_detector_counts` and `diag_09_qrs_overlay_12lead`; HR plot not covered. |
| `src/sqi_pipeline/diagnostics/qc_noisy_cases.py` | `{new_id}_wave12_clean_vs_noisy.png`, `{new_id}_snr_bar.png` | 12-lead clean/noisy overlays and per-lead SNR bars for sampled augmented cases. | Old `outputs/sqi/` output missing. | Partially covered by `diag_07_noise_allocation_snr` and final `fig_02_noise_generation_examples`; full 12-lead clean/noisy galleries are not covered. |
| `src/sqi_pipeline/diagnostics/qc_record84.py` | `hist_{sqi}.png`, `single_feature_auc_top20.png` | Pooled all-lead histograms by label and top-20 single-feature AUC. | Old `outputs/sqi/` output missing. | Histograms covered by `diag_04_pooled_sqi_distributions`; top-20 AUC is not covered. |
| `src/sqi_pipeline/diagnostics/qc_norm_record84_ks.py` | `raw_{lead}__sSQI_train_vs_test.png`, `norm_{lead}__sSQI_train_vs_test.png`, `raw_{lead}__kSQI_train_vs_test.png`, `norm_{lead}__kSQI_train_vs_test.png` | Train/test feature distribution before and after normalization. | Old `outputs/sqi/` output missing. | Covered by `diag_06_normalization_train_test`. |
| `src/sqi_pipeline/diagnostics/relabel_seta_stats.py` | `{sqi}_12lead_scatter.png`, `{lead}_7sqi_jitter.png`; also has record-metric jitter helper | Per-SQI across 12 leads and per-lead across 7 SQIs during relabel/stat exploration. | Old `outputs/sqi/` output missing. | Covered by `individual_sqi/` and `individual_lead/`; record-level single-metric jitter is not separately generated. |
| `src/sqi_pipeline/diagnostics/threshold_search.py` | `{rid}.png` under false-negative/error directories | 12-lead waveform panels for manually thresholded error cases. | Not present unless threshold-search is run. | Not covered; relevant only for manual-rule error analysis. |
| `src/sqi_pipeline/diagnostics/plot_baseline_record.py` | `baseline_{RECORD_ID}_{LEAD_NAME}_{SECONDS}s.png` | Single record/lead raw ECG plus 1 Hz low-pass baseline. | Old `outputs/sqi/` output missing. | Covered in polished form by final `fig_03_bassqi_examples`; not as the exact one-off record. |
| `src/sqi_pipeline/models/lm_mlp_search.py` | `val_metric_vs_J_seed{seed}.png`, `lm_mlp_roc_seed{seed}.png`, `lm_mlp_roc_maxacc_seed{seed}.png`, `lm_mlp_confmat_seed{seed}.png`, `{feature_set}_maxacc_seed{seed}.png` | MLP architecture selection, ROC/threshold diagnostics, confusion matrix, feature-set ROC sweeps. | Present in `outputs/sqi_paper_aligned/models/lm_mlp/`. | Not part of exploratory contact sheet; already available as model diagnostics. |
| `src/sqi_pipeline/models/svm_tables.py` | `{feature_set}_roc_maxacc_seed{seed}.png`, `Selected5_roc_maxacc_seed{seed}.png` | SVM ROC/threshold diagnostics for Table 5/6/7 feature sets. | Present in `outputs/sqi_paper_aligned/models/svm/roc/`. | Not part of exploratory contact sheet; already available as model diagnostics. |
| `src/sqi_pipeline/models/logreg_baseline.py` | `logreg_roc_seed{seed}.png`, `logreg_roc_maxacc_seed{seed}.png` | Logistic-regression baseline ROC diagnostics. | No current paper-aligned image found. | Not covered. |
| `src/sqi_pipeline/models/gnb_baseline.py` | `gnb_roc_seed{seed}.png`, `gnb_roc_maxacc_seed{seed}.png` | Gaussian Naive Bayes baseline ROC diagnostics. | No current paper-aligned image found. | Not covered. |
| `src/sqi_pipeline/diagnostics/plot_paper_figures.py` | `fig_01...fig_A1` as PDF and PNG | Final report figures and appendix warm-up validation. | Present in `reports/sqi_paper_aligned/figures/`. | Separate final-report set, not exploratory. |
| `src/sqi_pipeline/diagnostics/plot_exploratory_diagnostics.py` | `diag_01...diag_09`, `individual_sqi/*`, `individual_lead/*`, `diagnostic_contact_sheet.*` | Consolidated implementation-check diagnostics generated from paper-aligned artifacts. | Present in `reports/sqi_paper_aligned/exploratory_diagnostics/`. | This is the new consolidated set. |

## Gaps Worth Considering

These are the visualization code paths that are not fully represented by the current contact sheet:

1. Raw Set-a per-record `std/range` and raw mean PSD examples from `raw_qc_seta.py`.
2. Strict resampling alias/reference-filter error plots from `qc_resample_alias_check.py`.
3. Full 12-lead clean-vs-noisy galleries from `qc_noisy_cases.py`.
4. Top-20 single-feature AUC plot from `qc_record84.py`.
5. HR-by-lead detector boxplots from `qc_qrs_cache.py`.
6. Manual-threshold false-negative 12-lead panels from `threshold_search.py`.
7. GNB/LogReg baseline ROC diagnostics if those baselines are still part of the narrative.

For a report appendix, the most useful missing candidates are probably `single_feature_auc_top20`, `qc_resample_alias_check` summary, and a small clean-vs-noisy 12-lead gallery. The threshold-search panels are useful only if the report discusses manual SQI-rule failure cases.
