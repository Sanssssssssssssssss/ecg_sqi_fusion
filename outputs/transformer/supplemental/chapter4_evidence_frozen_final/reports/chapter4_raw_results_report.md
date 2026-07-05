# Chapter 4 Raw Results Report

This is a raw experiment report, not manuscript prose. Main sections use only current `chapter4_evidence_frozen_final` / official v116 sources; historical SQI paper-balanced artifacts are isolated in the appendix.

## 1. Run Manifest

| Field | Value |
| --- | --- |
| git commit | fbeecd3 |
| run date | 2026-07-02T14:45:30 |
| random seed | 0 |
| data policy | split first; train-only repair; validation/test original only |
| Set-A protocol path | outputs\transformer\supplemental\chapter4_evidence_frozen_final\seta_gapfill\data\protocol_gapfill.csv |
| BUT protocol path | E:\GPTProject2\ecg\outputs\transformer\v116_e31\analysis\good_medium_geometry_repair\clean_but_protocols\v116_gapfill_dual_goodorig_nm40_ms10_smc_s20260876 |
| output root | outputs\transformer\supplemental\chapter4_evidence_frozen_final |
| code command | python -m supplemental_transformer_experiments.chapter4_evidence.run --out chapter4_evidence_frozen_final pipeline --run |
| config file | CLI defaults; seed=0; Python/matplotlib figures |
| checkpoint path | Set-A: chapter4 output; BUT E31: query-mean fused v116 test_predictions.npz |

## 2. Current Protocol And Split Audit

Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/reports/protocol_audit.json`. BUT leakage checks use the official fold split, not the raw protocol metadata `split` column.

| Dataset | Split | Class | Original rows | Generated rows | Total rows | Generated in val/test |
| --- | --- | --- | --- | --- | --- | --- |
| Set-A | test | acceptable | 116 | 0 | 116 | 0 |
| Set-A | test | unacceptable | 33 | 0 | 33 | 0 |
| Set-A | train | acceptable | 541 | 0 | 541 | 0 |
| Set-A | train | unacceptable | 158 | 0 | 158 | 0 |
| Set-A | val | acceptable | 116 | 0 | 116 | 0 |
| Set-A | val | unacceptable | 34 | 0 | 34 | 0 |
| BUT | test | bad |  | 0 | 164 | 0 |
| BUT | test | good |  | 0 | 1053 | 0 |
| BUT | test | medium |  | 0 | 632 | 0 |
| BUT | train | bad |  |  | 8310 |  |
| BUT | train | good |  |  | 8310 |  |
| BUT | train | medium |  |  | 8310 |  |
| BUT | unused | bad |  |  | 1878 |  |
| BUT | unused | good |  |  | 114 |  |
| BUT | unused | medium |  |  | 970 |  |
| BUT | val | bad |  | 0 | 178 | 0 |
| BUT | val | good |  | 0 | 1053 | 0 |
| BUT | val | medium |  | 0 | 618 | 0 |

## 3. Current Set-A Data Repair Evidence

### Train-Poor Generated-vs-Original Distribution Diagnostics

Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/seta_distribution_repair_metrics.csv`. `generated_vs_original_c2st_auc` is a domain-separability metric, not model performance.

| construction | n_original | n_generated | generated_vs_original_c2st_auc | rbf_mmd | swd | cdf_gap_mean | cdf_gap_max | pca_overlap | pc1_var | pc2_var |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| within_original_resample | 158 | 383 | 0.7070 | 0.0003 | 0.0330 | 0.0233 | 0.0473 | 0.8172 | 0.5020 | 0.1723 |
| fixed_synthetic | 158 | 383 | 0.9695 | 0.5080 | 0.7186 | 0.5106 | 0.8420 | 0.8433 | 0.9092 | 0.0345 |
| quota_draw | 158 | 383 | 0.6446 | 0.0047 | 0.1263 | 0.2303 | 0.4594 | 0.9138 | 0.4661 | 0.2225 |
| smc_gapfill | 158 | 383 | 0.6173 | 0.0036 | 0.1099 | 0.2256 | 0.4557 | 0.9086 | 0.5749 | 0.1175 |

### Construction Source Audit

`fixed_synthetic` is the old paper `-6 dB em/ma` construction, not the current SMC generator. Transfer/recall comparisons use the selected-five SQI RBF-SVM source-only table below, matching the historical SQI supplemental route.

| construction | train_original_unacceptable_n | train_generated_unacceptable_n | train_generated_candidate_types | synthetic_source_contract | split_contract |
| --- | --- | --- | --- | --- | --- |
| native_imbalanced | 158 | 0 | {} | native_imbalanced | train-only generated; val/test original only |
| fixed_synthetic | 158 | 383 | {"paper_em": 192, "paper_ma": 191} | paper -6 dB em/ma | train-only generated; val/test original only |
| quota_draw | 158 | 383 | {"noise_style": 19, "ptb12_morph": 211, "seta_native_morph": 153} | quota_draw | train-only generated; val/test original only |
| smc_gapfill | 158 | 383 | {"noise_style": 19, "ptb12_morph": 211, "seta_native_morph": 153} | current SMC-selected pool | train-only generated; val/test original only |

### Paired MMD Calibration

Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/seta_paired_mmd_calibration.csv`.

| comparison | within_original_mmd_median | within_original_mmd_p05 | within_original_mmd_p95 | cross_mmd_median | cross_mmd_p05 | cross_mmd_p95 | delta_gt_0_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| within_original_resample | 0.0062 | 0.0031 | 0.0179 | 0.0064 | 0.0034 | 0.0195 | 0.5500 |
| fixed_synthetic | 0.0061 | 0.0030 | 0.0180 | 0.4489 | 0.3834 | 0.5088 | 1.0000 |
| quota_draw | 0.0061 | 0.0030 | 0.0180 | 0.0100 | 0.0065 | 0.0232 | 0.7700 |
| smc_gapfill | 0.0061 | 0.0030 | 0.0180 | 0.0088 | 0.0050 | 0.0237 | 0.7200 |

## 4. Current Set-A Model Comparison

Set-A model convention: paper-facing `Se` is original-unacceptable recall, `Sp` is acceptable recall, and `acceptable_positive_model_auc` treats acceptable as the positive class. Thresholds are selected on validation only. This frozen table was generated before the rename, so the explicit `acceptable_recall` and `original_unacceptable_recall` columns are the authoritative source for these old rows.

### Construction Effect: Source-Only

Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/seta_construction_source_only_models.csv`; displayed scope columns are regenerated from each arm's `splits/split.csv`. All rows use the same original-only validation (`116` acceptable, `34` unacceptable) and original-only held-out test (`116` acceptable, `33` unacceptable). For non-native construction arms, the classifier fit excludes the `158` original train-unacceptable rows and uses only the generated poor source, so paper `fixed_synthetic` is directly comparable with quota/SMC. The SQI normalization step remains the baseline train-only arm-level preprocessing; it uses no validation/test rows.

| run_id | construction | model | input | threshold_source | threshold | model_test_scope | train_acceptable_original_n | train_poor_source_n | source_only_train_poor_contract | train_original_unacceptable_n | train_generated_unacceptable_n | val_generated_rows | test_generated_rows | test_acceptable_n | test_unacceptable_n | acc | Se | Sp | acceptable_positive_model_auc | balanced_acc | acceptable_recall | original_unacceptable_recall | confusion | parameter_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seta_native_imbalanced_source_only_sqi_svmrbf_selected5 | native_imbalanced | SQI SVM-RBF selected5 | bSQI,basSQI,kSQI,sSQI,fSQI | validation_max_accuracy | 0.5515 | held-out original Set-A test only | 541 | 158 | original train unacceptable only | 158 | 0 | 0 | 0 | 116 | 33 | 0.9128 | 1.0000 | 0.6061 | 0.9062 | 0.8030 | 1.0000 | 0.6061 | {"tn": 20, "fp": 13, "fn": 0, "tp": 116} | paper_table6_fixed_C1_gamma0.14 |
| seta_fixed_synthetic_source_only_sqi_svmrbf_selected5 | fixed_synthetic | SQI SVM-RBF selected5 | bSQI,basSQI,kSQI,sSQI,fSQI | validation_max_accuracy | 0.2300 | held-out original Set-A test only | 541 | 383 | generated train unacceptable only; original train unacceptable excluded | 158 | 383 | 0 | 0 | 116 | 33 | 0.7785 | 0.9914 | 0.0303 | 0.5828 | 0.5108 | 0.9914 | 0.0303 | {"tn": 1, "fp": 32, "fn": 1, "tp": 115} | paper_table6_fixed_C1_gamma0.14 |
| seta_quota_draw_source_only_sqi_svmrbf_selected5 | quota_draw | SQI SVM-RBF selected5 | bSQI,basSQI,kSQI,sSQI,fSQI | validation_max_accuracy | 0.4400 | held-out original Set-A test only | 541 | 383 | generated train unacceptable only; original train unacceptable excluded | 158 | 383 | 0 | 0 | 116 | 33 | 0.8658 | 0.9397 | 0.6061 | 0.8890 | 0.7729 | 0.9397 | 0.6061 | {"tn": 20, "fp": 13, "fn": 7, "tp": 109} | paper_table6_fixed_C1_gamma0.14 |
| seta_smc_gapfill_source_only_sqi_svmrbf_selected5 | smc_gapfill | SQI SVM-RBF selected5 | bSQI,basSQI,kSQI,sSQI,fSQI | validation_max_accuracy | 0.4095 | held-out original Set-A test only | 541 | 383 | generated train unacceptable only; original train unacceptable excluded | 158 | 383 | 0 | 0 | 116 | 33 | 0.8658 | 0.9397 | 0.6061 | 0.8644 | 0.7729 | 0.9397 | 0.6061 | {"tn": 20, "fp": 13, "fn": 7, "tp": 109} | paper_table6_fixed_C1_gamma0.14 |

### Repaired Setup Model Comparison

Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/seta_repaired_model_comparison.csv`.

| run_id | construction | model | input | threshold_source | threshold | acc | Se | Sp | acceptable_positive_model_auc | balanced_acc | acceptable_recall | original_unacceptable_recall | confusion | parameter_source | val_threshold_acc | val_threshold_macro_f1 | val_threshold_acceptable_recall | val_threshold_unacceptable_recall | train_original_bad_recall_fixed05 | val_bad_recall_fixed05 | test_original_bad_recall_fixed05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seta_smc_gapfill_sqi_svmrbf_selected5 | smc_gapfill | SQI SVM-RBF selected5 | bSQI,basSQI,kSQI,sSQI,fSQI | validation_max_accuracy | 0.2045 | 0.8792 | 0.9569 | 0.6061 | 0.8814 | 0.7815 | 0.9569 | 0.6061 | {"tn": 20, "fp": 13, "fn": 5, "tp": 111} | paper_table6_fixed_C1_gamma0.14 |  |  |  |  |  |  |  |
| seta_smc_gapfill_sqi_svmrbf_all84 | smc_gapfill | SQI SVM-RBF all84 | all84 | validation_max_accuracy | 0.1820 | 0.8658 | 0.9397 | 0.6061 | 0.9310 | 0.7729 | 0.9397 | 0.6061 | {"tn": 20, "fp": 13, "fn": 7, "tp": 109} | paper_table6_fixed_C1_gamma0.14 |  |  |  |  |  |  |  |
| seta_smc_gapfill_lm_mlp_84 | smc_gapfill | SQI LM-MLP 84-4-1 | all84 | validation_max_accuracy | 0.7235 | 0.8993 | 0.9828 | 0.6061 | 0.8979 | 0.7944 | 0.9828 | 0.6061 | {"tn": 20, "fp": 13, "fn": 2, "tp": 114} |  |  |  |  |  |  |  |  |
| seta_smc_gapfill_e31style_waveform | smc_gapfill | 12-lead E31-style waveform comparator | 12-lead waveform-derived channels | validation_recall_balanced | 0.5905 | 0.9195 | 0.9483 | 0.8182 | 0.8879 | 0.8832 | 0.9483 | 0.8182 | {"tn": 27, "fp": 6, "fn": 6, "tp": 110} |  | 0.9000 | 0.8616 | 0.9224 | 0.8235 | 0.8101 | 0.8235 | 0.7879 |

## 5. Current BUT/v116 Evidence

### Data Cross-Checks

Source: protocol audit JSON plus official v116 fold split. The remembered low dual-AUC value is not cited because no traceable artifact has been found for this report.

| check | value | detail |
| --- | --- | --- |
| original BUT gap5 source | 18635 | {"bad": 1656, "good": 10530, "medium": 6449} |
| v116 final protocol | 31590 | {"bad": 10530, "good": 10530, "medium": 10530} |
| train exact balance | 8310/8310/8310 | {"bad": 8310, "good": 8310, "medium": 8310} |
| val/test generated rows | 0 | computed from official fold split only |
| official split source | outputs\transformer\v116_e31\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\rh_splits\v116_gapfill_dual_goodorig_nm40__k1_s20260876\fold0\original_region_atlas.csv | raw protocol metadata split column is not the leakage-audit split |
| allowed candidate types | but_native_morph, clean_style, original_but, ptb_morph | original_but, but_native_morph, ptb_morph, clean_style |
| nearest-neighbor leakage audit | 0 | sum of feature/raw near-duplicate counts across all scopes |
| E31 frozen test | acc=0.9486; macro-F1=0.9577 | good R=0.9430; medium R=0.9446; bad R=1.0000 |

### Candidate Composition

Source: v116 candidate-type counts from the official gap-fill report directory.

| class_name | v116_candidate_type | size | pct_within_class |
| --- | --- | --- | --- |
| bad | but_native_morph | 3514 | 33.3700 |
| bad | clean_style | 89 | 0.8500 |
| bad | original_but | 1656 | 15.7300 |
| bad | ptb_morph | 5271 | 50.0600 |
| good | original_but | 10530 | 100.0000 |
| medium | but_native_morph | 1616 | 15.3500 |
| medium | clean_style | 41 | 0.3900 |
| medium | original_but | 6449 | 61.2400 |
| medium | ptb_morph | 2424 | 23.0200 |

### V116 Dual Generated-vs-Original AUC Audit

Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/but_v116_dual_generated_auc_audit/dual_generated_auc.csv`. This is not an E31/SVM/MLP model score. It is a generated-vs-original domain audit for medium/bad using dual-view waveform summary features and `StandardScaler + LogisticRegression(C=0.5, class_weight='balanced')`; good is excluded because good has no generated rows.

| scope | status | rows | original_n | generated_n | generated_vs_original_domain_auc | symmetric_generated_vs_original_domain_auc | acc | ideal_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medium | ok | 9081 | 5000 | 4081 | 0.7098 | 0.7098 | 0.6858 | False |
| bad | ok | 6656 | 1656 | 5000 | 0.7090 | 0.7090 | 0.6193 | False |
| pooled_medium_bad_class_balanced | ok | 9156 | 4156 | 5000 | 0.7053 | 0.7053 | 0.6540 | False |

### Global Distribution Fit Diagnostics

Source: v116 global distribution metrics. `global_domain_auc_not_dual` is not the v116 dual audit and is not used for dual-AUC acceptance.

| scope | but_n | synthetic_n | rbf_mmd | sliced_wasserstein | quantile_loss | global_domain_auc_not_dual | pca_density_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_labels | 18635 | 31590 | 0.0596 | 0.3468 | 0.3355 | 0.7046 | 0.7757 |
| class_good | 10530 | 10530 | 0.0014 | 0.0556 | 0.0000 | 0.5100 | 1.0000 |
| class_medium | 6449 | 10530 | 0.0087 | 0.1175 | 0.0360 | 0.6491 | 0.9245 |
| class_bad | 1656 | 10530 | 0.0459 | 0.2294 | 0.1998 | 0.9336 | 0.7988 |

### Model Metrics

Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/but_model_comparison.csv` plus query-mean E31 `test_predictions.npz`. All rows are evaluated as `good/medium/bad` three-class models on the original-only BUT test split. The SQI rows below are legacy pseudo-12 compatibility artifacts and are superseded by the current single-lead 7-SQI/selected-five code path; do not cite them as Clifford 12-lead 84-SQI results. `Se/Sp` is not used in this multiclass table; class recalls are reported directly.

| run_id | model | input | task | test_acc | test_macro_f1 | good_recall | intermediate_recall | poor_recall | poor_fpr_nonpoor | poor_vs_rest_auc | collapsed_good_vs_rest_acc | collapsed_good_vs_rest_auc | collapsed_good_vs_rest_confusion | confusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_svm_rbf_legacy_pseudo12_sqi_multiclass | Legacy SQI SVM-RBF pseudo-12 | legacy single-lead SQI duplicated to pseudo-12 84-SQI; superseded by current single-lead 7-SQI code | good/medium/bad | 0.7999 | 0.8326 | 0.7892 | 0.7753 | 0.9634 | 0.0071 | 0.9976 | 0.8075 | 0.9017 | {"tn": 662, "fp": 134, "fn": 222, "tp": 831} | [[831, 219, 3], [133, 490, 9], [1, 5, 158]] |
| but_svm_rbf_selected5_legacy_pseudo12_multiclass | Legacy SQI SVM-RBF selected5 pseudo-12 | legacy selected-five SQI duplicated to pseudo-12; superseded by current single-lead selected5 code | good/medium/bad | 0.7756 | 0.8093 | 0.7597 | 0.7532 | 0.9634 | 0.0113 | 0.9987 | 0.7853 | 0.8811 | {"tn": 652, "fp": 144, "fn": 253, "tp": 800} | [[800, 246, 7], [144, 476, 12], [0, 6, 158]] |
| but_lm_mlp_legacy_pseudo12_sqi_ovr_multiclass | Legacy SQI LM-MLP pseudo-12 84-8-1 OvR | legacy single-lead SQI duplicated to pseudo-12 84-SQI; superseded by current single-lead 7-SQI code | good/medium/bad | 0.8102 | 0.8434 | 0.8110 | 0.7658 | 0.9756 | 0.0053 | 0.9990 | 0.8150 | 0.9004 | {"tn": 653, "fp": 143, "fn": 199, "tp": 854} | [[854, 195, 4], [143, 484, 5], [0, 4, 160]] |
| but_e31_query_mean_fused_conformer | E31 query-mean fused Conformer | 8-channel waveform-derived time series | good/medium/bad | 0.9486 | 0.9577 | 0.9430 | 0.9446 | 1.0000 | 0.0018 | 1.0000 | 0.9502 | 0.9889 | {"tn": 764, "fp": 32, "fn": 60, "tp": 993} | [[993, 60, 0], [32, 597, 3], [0, 0, 164]] |

### Good--Medium Boundary Audit

Source: `outputs/transformer/supplemental/chapter4_evidence_frozen_final/tables/but_good_medium_boundary_audit.csv`. This audit uses the same original-only BUT test split as the model table. `boundary_exchange_errors` is exactly `good_to_medium + medium_to_good`; bad-related columns are shown only to verify that the main error reduction is concentrated at the good/medium boundary.

| model | good_to_medium | good_to_bad | medium_to_good | bad_to_good | boundary_exchange_errors | good_medium_test_n | boundary_exchange_rate | confusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Legacy SQI SVM-RBF pseudo-12 | 219 | 3 | 133 | 1 | 352 | 1685 | 0.2089 | [[831, 219, 3], [133, 490, 9], [1, 5, 158]] |
| Legacy SQI LM-MLP pseudo-12 84-8-1 OvR | 195 | 4 | 143 | 0 | 338 | 1685 | 0.2006 | [[854, 195, 4], [143, 484, 5], [0, 4, 160]] |
| Conformer | 60 | 0 | 32 | 0 | 92 | 1685 | 0.0546 | [[993, 60, 0], [32, 597, 3], [0, 0, 164]] |

## 6. Figure Index

Every figure has source-data CSV under `outputs/transformer/supplemental/chapter4_evidence_frozen_final/figures/source_data/`; the audit checks the current Chapter 4 figure/source-data paths.

| Figure | File path | Source data | Conclusion role |
| --- | --- | --- | --- |
| fig_D1_distribution_repair_summary | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_D1_distribution_repair_summary.png | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\source_data | raw evidence |
| fig_D2_top_drift_features | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_D2_top_drift_features.png | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\source_data | raw evidence |
| fig_D3_seta_ours_vs_paper_em_ma_distribution | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_D3_seta_ours_vs_paper_em_ma_distribution.png | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\source_data | raw evidence |
| fig_M1_seta_model_performance | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_M1_seta_model_performance.png | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\source_data | raw evidence |
| fig_D4_but_medium_bad_gapfill_distribution | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_D4_but_medium_bad_gapfill_distribution.png | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\source_data | raw evidence |
| fig_M2_but_model_comparison | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_M2_but_model_comparison.png | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\source_data | raw evidence |
| fig_M3_but_good_medium_boundary_audit | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_M3_but_good_medium_boundary_audit.png | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\source_data | raw evidence |
| fig_M4_but_mlp_error_conformer_correct_examples | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_M4_but_mlp_error_conformer_correct_examples.png | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\source_data | raw evidence |
| fig_M5_but_query_patching | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_M5_but_query_patching.png | outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\source_data | raw evidence |

## 7. Historical SQI Paper-Balanced Comparator Appendix

The following tables replay `outputs/sqi_supplemental/existing_seed0`. They are historical paper-balanced artifacts, not the current train-only Set-A SMC protocol.

### Historical existing_seed0 Synthetic Domain Shift

Paper synthetic poor here means `-6 dB em` and `-6 dB ma` from the SQI baseline supplemental line.

| feature_set | comparison | domain_auc | ci_low | ci_high | n_original_poor | n_synthetic_poor | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 84-SQI | original poor vs synthetic poor | 0.9735 |  |  | 225 | 548 | SQI baseline final_claims/domain_shift_metrics.csv |
| 84-SQI | RBF-MMD2 original poor vs synthetic poor |  |  |  | 225 | 548 | MMD2=0.330871; permutation p=0.000999001 |
| paper_waveform_all_103 | original bad vs synthetic poor | 0.9800 | 0.9646 | 0.9920 | 225 | 548 | SQI supplemental waveform_domain_auc/paper_waveform_all_domain_auc.csv |
| paper_waveform_all_103 | original bad vs synthetic em | 0.9823 | 0.9696 | 0.9912 | 225 | 274 | SQI supplemental waveform_domain_auc/paper_waveform_all_domain_auc.csv |
| paper_waveform_all_103 | original bad vs synthetic ma | 0.9861 | 0.9719 | 0.9963 | 225 | 274 | SQI supplemental waveform_domain_auc/paper_waveform_all_domain_auc.csv |

### Historical existing_seed0 Cross-Domain Performance

Rows correspond to the old-paper `original poor`, `-6 dB em`, and `-6 dB ma` poor-domain transfer matrix.

| train_poor_domain | test_poor_domain | train_n | val_n | test_n | threshold | test_Ac | Se | Sp | acceptable_positive_model_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original poor | original poor | 699 | 150 | 149 | 0.6395 | 0.8993 | 0.9741 | 0.6364 | 0.9120 |
| original poor | em | 699 | 150 | 157 | 0.6395 | 0.7389 | 0.9741 | 0.0732 | 0.9331 |
| original poor | ma | 699 | 150 | 157 | 0.6395 | 0.9363 | 0.9741 | 0.8293 | 0.9853 |
| em | original poor | 733 | 157 | 149 | 0.5535 | 0.7919 | 1.0000 | 0.0606 | 0.4313 |
| em | em | 733 | 157 | 157 | 0.5535 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| em | ma | 733 | 157 | 157 | 0.5535 | 0.9809 | 1.0000 | 0.9268 | 0.9994 |
| ma | original poor | 733 | 157 | 149 | 0.0845 | 0.7785 | 1.0000 | 0.0000 | 0.4323 |
| ma | em | 733 | 157 | 157 | 0.0845 | 0.8535 | 1.0000 | 0.4390 | 0.9998 |
| ma | ma | 733 | 157 | 157 | 0.0845 | 0.9873 | 1.0000 | 0.9512 | 1.0000 |
| synthetic poor | original poor | 925 | 198 | 149 | 0.2240 | 0.7919 | 1.0000 | 0.0606 | 0.4287 |
| synthetic poor | em | 925 | 198 | 157 | 0.2240 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| synthetic poor | ma | 925 | 198 | 157 | 0.2240 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Historical existing_seed0 Selected-Five Aggregate

| model | protocol | input | threshold | source | Ac | Ac_ci_low | Ac_ci_high | Se | Se_ci_low | Se_ci_high | Sp | Sp_ci_low | Sp_ci_high | acceptable_positive_model_auc | acceptable_positive_model_auc_ci_low | acceptable_positive_model_auc_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP selected-five | SQI supplemental paper-balanced test | selected-five 12-lead SQI | 0.6295 | outputs/sqi_supplemental/existing_seed0/model_diagnostics/selected5_source_bootstrap_metrics.csv | 0.9177 | 0.8791 | 0.9528 | 0.9310 | 0.8829 | 0.9748 | 0.9043 | 0.8421 | 0.9533 | 0.9552 | 0.9186 | 0.9846 |
| SVM selected-five | SQI supplemental paper-balanced test | selected-five 12-lead SQI | 0.5215 | outputs/sqi_supplemental/existing_seed0/model_diagnostics/selected5_source_bootstrap_metrics.csv | 0.9394 | 0.9041 | 0.9671 | 0.9655 | 0.9273 | 0.9917 | 0.9130 | 0.8529 | 0.9609 | 0.9615 | 0.9306 | 0.9851 |

### Historical existing_seed0 Selected-Five Subgroup Recall

For unacceptable groups, `group_recall` is rejection/poor recall. This table is the correct place to compare old SQI supplemental subgroup results, not the current `fixed_synthetic` row.

| model | sample_group | n | group_recall_metric | group_recall | acceptance_rate | rejection_rate | threshold | score_mean | score_median | pairwise_model_auc | source | label_acceptable | score_q10 | score_q90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP selected-five | original acceptable | 116 | acceptable_recall | 0.9310 | 0.9310 | 0.0690 | 0.6295 | 0.9352 | 0.9999 |  | outputs/sqi_supplemental/existing_seed0/model_diagnostics/stratified_score_summary.csv | 1 | 0.9044 | 1.0000 |
| MLP selected-five | original unacceptable | 33 | poor_recall | 0.6667 | 0.3333 | 0.6667 | 0.6295 | 0.3300 | 0.0001 |  | outputs/sqi_supplemental/existing_seed0/model_diagnostics/stratified_score_summary.csv | 0 | 0.0000 | 0.9994 |
| MLP selected-five | synthetic em | 41 | poor_recall | 1.0000 | 0.0000 | 1.0000 | 0.6295 | 0.0088 | 0.0002 |  | outputs/sqi_supplemental/existing_seed0/model_diagnostics/stratified_score_summary.csv | 0 | 0.0000 | 0.0056 |
| MLP selected-five | synthetic ma | 41 | poor_recall | 1.0000 | 0.0000 | 1.0000 | 0.6295 | 0.0002 | 0.0000 |  | outputs/sqi_supplemental/existing_seed0/model_diagnostics/stratified_score_summary.csv | 0 | 0.0000 | 0.0004 |
| SVM selected-five | original acceptable | 116 | acceptable_recall | 0.9655 | 0.9655 | 0.0345 | 0.5215 | 0.9043 | 0.9616 |  | outputs/sqi_supplemental/existing_seed0/model_diagnostics/stratified_score_summary.csv | 1 | 0.7163 | 0.9929 |
| SVM selected-five | original unacceptable | 33 | poor_recall | 0.6970 | 0.3030 | 0.6970 | 0.5215 | 0.3328 | 0.0996 |  | outputs/sqi_supplemental/existing_seed0/model_diagnostics/stratified_score_summary.csv | 0 | 0.0117 | 0.9464 |
| SVM selected-five | synthetic em | 41 | poor_recall | 1.0000 | 0.0000 | 1.0000 | 0.5215 | 0.0552 | 0.0389 |  | outputs/sqi_supplemental/existing_seed0/model_diagnostics/stratified_score_summary.csv | 0 | 0.0100 | 0.1101 |
| SVM selected-five | synthetic ma | 41 | poor_recall | 1.0000 | 0.0000 | 1.0000 | 0.5215 | 0.0163 | 0.0073 |  | outputs/sqi_supplemental/existing_seed0/model_diagnostics/stratified_score_summary.csv | 0 | 0.0031 | 0.0434 |

### Historical existing_seed0 Cross-Noise Generalization

The often-confusing `0.4287` value is the `synthetic_poor_to_original_poor` acceptable-positive model AUC in this table; its original-poor recall is `0.0606` under the current paper-facing `Se` convention.

| scenario | train_n | val_n | test_n | threshold | test_Ac | Se | Sp | acceptable_positive_model_auc | test_tn | test_fp | test_fn | test_tp | source | sqis | val_Ac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_em_test_ma | 733 | 157 | 157 | 0.5535 | 0.9809 | 1.0000 | 0.9268 | 0.9994 | 38 | 3 | 0 | 116 | outputs/sqi_supplemental/existing_seed0/generalization/cross_noise_generalization_svm.csv | bSQI,basSQI,kSQI,sSQI,fSQI | 1.0000 |
| train_ma_test_em | 733 | 157 | 157 | 0.0845 | 0.8535 | 1.0000 | 0.4390 | 0.9998 | 18 | 23 | 0 | 116 | outputs/sqi_supplemental/existing_seed0/generalization/cross_noise_generalization_svm.csv | bSQI,basSQI,kSQI,sSQI,fSQI | 1.0000 |
| synthetic_poor_to_original_poor | 925 | 198 | 149 | 0.2240 | 0.7919 | 1.0000 | 0.0606 | 0.4287 | 2 | 31 | 0 | 116 | outputs/sqi_supplemental/existing_seed0/generalization/cross_noise_generalization_svm.csv | bSQI,basSQI,kSQI,sSQI,fSQI | 0.9949 |
| original_poor_to_synthetic_poor | 699 | 150 | 198 | 0.6395 | 0.7576 | 0.9741 | 0.4512 | 0.9592 | 37 | 45 | 3 | 113 | outputs/sqi_supplemental/existing_seed0/generalization/cross_noise_generalization_svm.csv | bSQI,basSQI,kSQI,sSQI,fSQI | 0.9467 |

## 8. Audit Notes

| item | status | note |
| --- | --- | --- |
| AUC naming | fixed | Main report avoids generic `AUC`; columns are named as model, domain, transfer, or poor-vs-rest AUC. |
| BUT 0.9336 | scoped | `0.9336` is `global_domain_auc_not_dual` for class_bad, not v116 dual AUC. |
| BUT remembered low dual-AUC | not used | No traceable artifact found in the audited current line; current v116 dual AUC remains medium 0.7098, bad 0.7090, pooled 0.7053. |
| historical SQI paper-balanced | isolated | All `outputs/sqi_supplemental/existing_seed0` rows are appendix-only. |

## 9. Candidate Diagnostics For Observation Section

| Diagnostic | Trigger | Needed output | Decision |
| --- | --- | --- | --- |
| source sensitivity | C2ST high or source imbalance remains | score by source, embedding C2ST | defer until raw evidence readout |
| shortcut check | generated source remains separable | provenance classifier | defer |
| local evidence maps | waveform model improves boundary recall | ECG + local map overlays | defer |
| input ablation | need to explain waveform gain | channel ablation table | defer |
| calibration | models close or threshold-sensitive | ECE, reliability, threshold sweep | defer |
