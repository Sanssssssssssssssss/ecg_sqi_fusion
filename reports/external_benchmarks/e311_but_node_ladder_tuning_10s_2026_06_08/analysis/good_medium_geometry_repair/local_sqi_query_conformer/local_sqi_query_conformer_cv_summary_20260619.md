# LocalSQI Query Conformer CV Summary

- Created: 2026-06-20 00:01:58
- Scope: external-only LocalSQI Query Conformer experiments under the E3.11f clean BUT protocol.
- Policy: `margin_ge_5s_drop_outlier`; 5-fold record-heldout GroupKFold; selection/evaluation here is clean protocol CV, not original BUT selection.
- Training budget: quick 1-2 epoch CV per candidate, intended for architecture triage rather than final convergence.

## Headline
Best mean clean-test macro-F1 in this quick CV is `A0_nohi_noquery` from `stage_a`: acc `0.9190`, macro-F1 `0.7904`, recalls good/medium/bad `0.9129/0.9119/0.6000`.
Best mean bad recall is `A0_nohi_noquery` from `stage_a` with bad recall `0.6000` and macro-F1 `0.7904`.
The strongest lesson is not that a larger head wins. Query readout helps in some folds, but current local-map supervision and hierarchical bad-first heads are fold-unstable. Several candidates recover flatline/baseline/detail features well, while detector agreement and qrs visibility remain inconsistent across folds.

## Stage Winners By Mean Clean-Test Macro-F1
| stage | candidate | acc_mean | macro_f1_mean | good_recall_mean | medium_recall_mean | bad_recall_mean | acc_min | macro_f1_min | bad_fpr_nonbad_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sqi_fusion_ladder | L1_oracle_sqi_diag | 0.9391 | 0.7516 | 0.9712 | 0.9227 | 0.4000 | 0.8295 | 0.5480 | 0.0002 |
| stage_a | A0_nohi_noquery | 0.9190 | 0.7904 | 0.9129 | 0.9119 | 0.6000 | 0.7762 | 0.6199 | 0.0008 |
| stage_b | B3_query_hier_local | 0.9054 | 0.7113 | 0.9050 | 0.9291 | 0.4000 | 0.7998 | 0.5206 | 0.0004 |
| stage_c | C0_query_hier_local | 0.8567 | 0.6761 | 0.8447 | 0.9163 | 0.4000 | 0.7279 | 0.4760 | 0.0002 |
| stage_d | D2_masked_factor_pretrain | 0.9007 | 0.7173 | 0.8837 | 0.9511 | 0.4000 | 0.6879 | 0.4516 | 0.0004 |

## Top Candidates Overall
| stage | candidate | acc_mean | macro_f1_mean | good_recall_mean | medium_recall_mean | bad_recall_mean | worst_recall_mean | acc_min | macro_f1_min | bad_fpr_nonbad_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage_a | A0_nohi_noquery | 0.9190 | 0.7904 | 0.9129 | 0.9119 | 0.6000 | 0.6000 | 0.7762 | 0.6199 | 0.0008 |
| sqi_fusion_ladder | L1_oracle_sqi_diag | 0.9391 | 0.7516 | 0.9712 | 0.9227 | 0.4000 | 0.4000 | 0.8295 | 0.5480 | 0.0002 |
| stage_a | A1_hi_noquery | 0.8642 | 0.7493 | 0.8502 | 0.9347 | 0.6000 | 0.6000 | 0.7256 | 0.5560 | 0.0002 |
| sqi_fusion_ladder | L2_pred_sqi_stopgrad | 0.9185 | 0.7198 | 0.9192 | 0.9308 | 0.4000 | 0.4000 | 0.8492 | 0.5524 | 0.0002 |
| stage_d | D2_masked_factor_pretrain | 0.9007 | 0.7173 | 0.8837 | 0.9511 | 0.4000 | 0.4000 | 0.6879 | 0.4516 | 0.0004 |
| stage_b | B3_query_hier_local | 0.9054 | 0.7113 | 0.9050 | 0.9291 | 0.4000 | 0.4000 | 0.7998 | 0.5206 | 0.0004 |
| stage_b | B2_query_hier_nolocal | 0.9025 | 0.7078 | 0.8994 | 0.9318 | 0.4000 | 0.4000 | 0.8174 | 0.5320 | 0.0003 |
| stage_a | A2_nohi_query | 0.8968 | 0.7066 | 0.8937 | 0.9315 | 0.4000 | 0.4000 | 0.7527 | 0.4912 | 0.0002 |
| stage_b | B1_query_ce_local | 0.8645 | 0.7059 | 0.9426 | 0.9309 | 0.4000 | 0.4000 | 0.5839 | 0.4660 | 0.0007 |
| stage_b | B0_query_ce_nolocal | 0.8745 | 0.6915 | 0.8568 | 0.9556 | 0.4000 | 0.4000 | 0.7785 | 0.5104 | 0.0003 |
| stage_d | D0_no_pretrain | 0.8521 | 0.6783 | 0.8164 | 0.9588 | 0.4000 | 0.4000 | 0.5531 | 0.3688 | 0.0002 |
| stage_c | C0_query_hier_local | 0.8567 | 0.6761 | 0.8447 | 0.9163 | 0.4000 | 0.4000 | 0.7279 | 0.4760 | 0.0002 |
| stage_d | D1_masked_recon_pretrain | 0.8450 | 0.6736 | 0.8095 | 0.9639 | 0.4000 | 0.4000 | 0.6848 | 0.4522 | 0.0003 |
| sqi_fusion_ladder | L3_pred_sqi_e2e | 0.8353 | 0.6178 | 0.9112 | 0.9253 | 0.2000 | 0.2000 | 0.5742 | 0.4576 | 0.0004 |
| sqi_fusion_ladder | L0_waveform_only | 0.7824 | 0.5918 | 0.8363 | 0.9186 | 0.2000 | 0.2000 | 0.5688 | 0.3997 | 0.0005 |
| stage_a | A3_hi_query | 0.7768 | 0.5797 | 0.8137 | 0.9456 | 0.2000 | 0.2000 | 0.5740 | 0.4543 | 0.0000 |

## Key Feature Recovery For Top Candidates
| stage | candidate | qrs_visibility_corr_mean | qrs_band_ratio_corr_mean | template_corr_corr_mean | baseline_step_corr_mean | flatline_ratio_corr_mean | non_qrs_diff_p95_corr_mean | sqi_basSQI_corr_mean | detector_agreement_corr_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage_a | A0_nohi_noquery | 0.3759 | 0.0429 | 0.4435 | 0.7254 | 0.8264 | 0.8617 | 0.2884 | -0.3328 |
| sqi_fusion_ladder | L1_oracle_sqi_diag | 0.3500 | 0.0686 | 0.4215 | 0.7265 | 0.8139 | 0.8435 | 0.4851 | -0.2739 |
| stage_a | A1_hi_noquery | 0.3776 | 0.2124 | 0.4881 | 0.7293 | 0.8531 | 0.8543 | 0.5284 | -0.0556 |
| sqi_fusion_ladder | L2_pred_sqi_stopgrad | 0.4148 | -0.0235 | 0.4477 | 0.7383 | 0.8375 | 0.8688 | 0.4356 | 0.1323 |
| stage_d | D2_masked_factor_pretrain | 0.4166 | 0.0832 | 0.5041 | 0.7765 | 0.8591 | 0.8444 | 0.6267 | 0.1042 |
| stage_b | B3_query_hier_local | 0.4015 | 0.1128 | 0.4586 | 0.7490 | 0.8534 | 0.8727 | 0.3295 | -0.1778 |
| stage_b | B2_query_hier_nolocal | 0.4230 | -0.0471 | 0.4814 | 0.6987 | 0.8788 | 0.8799 | 0.3507 | -0.2415 |
| stage_a | A2_nohi_query | 0.3837 | -0.0427 | 0.4338 | 0.7231 | 0.8395 | 0.8734 | 0.3696 | 0.1809 |
| stage_b | B1_query_ce_local | 0.3763 | -0.1803 | 0.4887 | 0.6319 | 0.8485 | 0.8752 | 0.1539 | 0.0275 |
| stage_b | B0_query_ce_nolocal | 0.3966 | -0.0249 | 0.4755 | 0.7348 | 0.8821 | 0.8754 | 0.5886 | -0.0176 |

## Gradient Audit Summary
| loss_a | loss_b | cosine_mean | cosine_min | cosine_max | grad_norm_a_mean | grad_norm_b_mean |
| --- | --- | --- | --- | --- | --- | --- |
| class | artifact | -0.5933 | -0.9840 | -0.0508 | 0.9741 | 2.0487 |
| class | factor | -0.0222 | -0.0555 | 0.0171 | 0.9741 | 0.4731 |
| factor | local | 0.0000 | 0.0000 | 0.0000 | 0.4731 | 1.4029 |
| class | local | 0.0000 | 0.0000 | 0.0000 | 0.9741 | 1.4029 |
| local | artifact | 0.0000 | 0.0000 | 0.0000 | 1.4029 | 2.0487 |
| factor | artifact | 0.0683 | 0.0385 | 0.1039 | 0.4731 | 2.0487 |
| class | class | 1.0000 | 1.0000 | 1.0000 | 0.9741 |  |
| artifact | artifact | 1.0000 | 1.0000 | 1.0000 | 2.0487 |  |
| factor | factor | 1.0000 | 1.0000 | 1.0000 | 0.4731 |  |
| local | local | 1.0000 | 1.0000 | 1.0000 | 1.4029 |  |

## Interpretation
- Stage A: A0/A1/A2 are safer than A3 in mean CV. Query readout helps in some folds, but adding the high-resolution local path plus current local pseudo-targets can destabilize bad recall.
- Stage B/C: Hierarchical heads are conceptually clean, but the current local supervision is not reliable enough to make B3/C0 dominate. The gradient audit also shows strong conflict between class and artifact losses, so PCGrad or gradient surgery is justified only there, not as blanket complexity.
- SQI fusion ladder: oracle/predicted SQI variants are useful diagnostics. Predicted SQI stopgrad is competitive, but does not automatically solve bad recall. The encoder still needs better QRS/detector/baseline facts, not just a stronger downstream classifier.
- Stage D: masked pretraining/factor pretraining is mechanically viable, but one short pretrain epoch is not enough to win. D2 improves some basSQI/qrs-band recovery in selected folds, so longer safer pretraining is worth a next pass.

## Artifact Locations
- Runner code: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_local_sqi_query_conformer.py`
- Report directory: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\local_sqi_query_conformer`
- Output/run directory with checkpoints and logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\local_sqi_query_conformer`
- Record-heldout splits: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\local_sqi_query_conformer\recordheldout_splits\margin_ge_5s_drop_outlier_groupkfold5_seed20260619`
- Full CV log: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\local_sqi_query_conformer\cv_full_20260619_230427.log`
- All fold metrics CSV: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\local_sqi_query_conformer\local_sqi_cv_metrics_all.csv`
- Clean-test aggregate CSV: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\local_sqi_query_conformer\local_sqi_cv_clean_test_summary.csv`
- Feature recovery long aggregate CSV: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\local_sqi_query_conformer\local_sqi_cv_feature_recovery_summary.csv`
- Key feature pivot CSV: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\local_sqi_query_conformer\local_sqi_cv_feature_corr_pivot.csv`
- Gradient audit summary CSV: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\local_sqi_query_conformer\local_sqi_cv_gradient_audit_summary.csv`

## Notes For Next Round
- Do not claim route/rule artifacts as the formal model here. These experiments keep waveform-only inference; oracle SQI is diagnostic.
- Next useful work: improve local target reliability for QRS/detector agreement, add gradient conflict handling only for class-vs-artifact conflict, and run longer D2-style pretraining only if per-class feature recovery improves without bad collapse.
- Because this was quick CV, final candidate confirmation still needs longer training and original bucketed report-only evaluation after clean-CV selection.
