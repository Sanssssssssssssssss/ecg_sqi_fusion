# SQI-Aware Waveform Transformer

Input at inference is waveform-derived tokens only. The 47 SQI/geometry columns are teacher targets during PTB/synthetic training and are not provided to the classifier or to BUT evaluation.

## Main Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m | Key Aux MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sqiquery_gm_focus_raw | synthetic_test | 0.992824 | 0.991460 | 0.983264 | 0.999188 | 0.979253 | 8 | 1 | 0 | 5 | 0.7211 |
| sqiquery_gm_focus_raw_badcal | synthetic_test | 0.993337 | 0.992299 | 0.983264 | 0.999188 | 0.983402 | 8 | 1 | 0 | 4 | nan |
| sqiquery_gm_focus_raw | original_test_all_10s+ | 0.792497 | 0.606351 | 0.667582 | 0.957524 | 0.121655 | 1210 | 188 | 0 | 298 | nan |
| sqiquery_gm_focus_raw_badcal | original_test_all_10s+ | 0.793441 | 0.617762 | 0.667582 | 0.957298 | 0.143552 | 1210 | 188 | 0 | 289 | nan |
| sqiquery_gm_focus_raw | original_all_10s+ | 0.841152 | 0.863215 | 0.773221 | 0.913718 | 0.914286 | 3865 | 917 | 0 | 390 | nan |
| sqiquery_gm_focus_raw_badcal | original_all_10s+ | 0.841425 | 0.863618 | 0.773221 | 0.913624 | 0.916178 | 3865 | 917 | 0 | 380 | nan |
| sqiquery_gm_focus_raw | bad_core_nearboundary | 0.420168 | 0.197239 | 0.000000 | 0.000000 | 0.420168 | 0 | 0 | 0 | 69 | nan |
| sqiquery_gm_focus_raw_badcal | bad_core_nearboundary | 0.495798 | 0.220974 | 0.000000 | 0.000000 | 0.495798 | 0 | 0 | 0 | 60 | nan |
| sqiquery_gm_focus_raw | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 229 | nan |
| sqiquery_gm_focus_raw_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 229 | nan |
| sqiquery_badstress_hardneg_raw | synthetic_test | 0.988724 | 0.981823 | 0.991632 | 0.989448 | 0.979253 | 1 | 0 | 0 | 5 | 0.7187 |
| sqiquery_badstress_hardneg_raw_badcal | synthetic_test | 0.988724 | 0.981823 | 0.991632 | 0.989448 | 0.979253 | 1 | 0 | 0 | 5 | nan |
| sqiquery_badstress_hardneg_raw | original_test_all_10s+ | 0.272974 | 0.276492 | 0.166484 | 0.314957 | 0.763990 | 274 | 58 | 0 | 59 | nan |
| sqiquery_badstress_hardneg_raw_badcal | original_test_all_10s+ | 0.272974 | 0.276492 | 0.166484 | 0.314957 | 0.763990 | 274 | 58 | 0 | 59 | nan |
| sqiquery_badstress_hardneg_raw | original_all_10s+ | 0.262502 | 0.259557 | 0.054451 | 0.243320 | 0.971996 | 513 | 76 | 0 | 110 | nan |
| sqiquery_badstress_hardneg_raw_badcal | original_all_10s+ | 0.262502 | 0.259557 | 0.054451 | 0.243320 | 0.971996 | 513 | 76 | 0 | 110 | nan |
| sqiquery_badstress_hardneg_raw | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 | nan |
| sqiquery_badstress_hardneg_raw_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 | nan |
| sqiquery_badstress_hardneg_raw | bad_outlier_stress | 0.667808 | 0.266940 | 0.000000 | 0.000000 | 0.667808 | 0 | 0 | 0 | 59 | nan |
| sqiquery_badstress_hardneg_raw_badcal | bad_outlier_stress | 0.667808 | 0.266940 | 0.000000 | 0.000000 | 0.667808 | 0 | 0 | 0 | 59 | nan |
| sqiquery_multiview_robust3 | synthetic_test | 0.987186 | 0.981215 | 0.972803 | 0.994318 | 0.979253 | 7 | 2 | 0 | 5 | 0.7133 |
| sqiquery_multiview_robust3_badcal | synthetic_test | 0.987186 | 0.981215 | 0.972803 | 0.994318 | 0.979253 | 7 | 2 | 0 | 5 | nan |
| sqiquery_multiview_robust3 | original_test_all_10s+ | 0.558334 | 0.449688 | 0.397527 | 0.711026 | 0.338200 | 571 | 719 | 0 | 124 | nan |
| sqiquery_multiview_robust3_badcal | original_test_all_10s+ | 0.558334 | 0.449688 | 0.397527 | 0.711026 | 0.338200 | 571 | 719 | 0 | 124 | nan |
| sqiquery_multiview_robust3 | original_all_10s+ | 0.603138 | 0.622696 | 0.351875 | 0.847102 | 0.922800 | 8820 | 877 | 0 | 259 | nan |
| sqiquery_multiview_robust3_badcal | original_all_10s+ | 0.603138 | 0.622696 | 0.351875 | 0.847102 | 0.922800 | 8820 | 877 | 0 | 259 | nan |
| sqiquery_multiview_robust3 | bad_core_nearboundary | 0.193277 | 0.107981 | 0.000000 | 0.000000 | 0.193277 | 0 | 0 | 0 | 96 | nan |
| sqiquery_multiview_robust3_badcal | bad_core_nearboundary | 0.193277 | 0.107981 | 0.000000 | 0.000000 | 0.193277 | 0 | 0 | 0 | 96 | nan |
| sqiquery_multiview_robust3 | bad_outlier_stress | 0.397260 | 0.189542 | 0.000000 | 0.000000 | 0.397260 | 0 | 0 | 0 | 28 | nan |
| sqiquery_multiview_robust3_badcal | bad_outlier_stress | 0.397260 | 0.189542 | 0.000000 | 0.000000 | 0.397260 | 0 | 0 | 0 | 28 | nan |

## Teacher Recovery

| Candidate | Feature | MAE(z) | Corr |
|---|---|---:|---:|
| sqiquery_badstress_hardneg_raw | pca_margin | 0.3356 | 0.7436 |
| sqiquery_badstress_hardneg_raw | pc1 | 0.3899 | 0.7302 |
| sqiquery_badstress_hardneg_raw | non_qrs_diff_p95 | 0.4072 | 0.7184 |
| sqiquery_badstress_hardneg_raw | sqi_bSQI | 0.4194 | 0.6204 |
| sqiquery_badstress_hardneg_raw | boundary_confidence | 0.6383 | 0.3291 |
| sqiquery_badstress_hardneg_raw | template_corr | 0.6389 | 0.4441 |
| sqiquery_badstress_hardneg_raw | low_amp_ratio | 0.6675 | 0.3679 |
| sqiquery_badstress_hardneg_raw | detector_agreement | 0.6800 | 0.2259 |
| sqiquery_gm_focus_raw | pca_margin | 0.3274 | 0.7506 |
| sqiquery_gm_focus_raw | pc1 | 0.3903 | 0.7334 |
| sqiquery_gm_focus_raw | sqi_bSQI | 0.4119 | 0.6290 |
| sqiquery_gm_focus_raw | non_qrs_diff_p95 | 0.4211 | 0.7195 |
| sqiquery_gm_focus_raw | template_corr | 0.6429 | 0.4562 |
| sqiquery_gm_focus_raw | boundary_confidence | 0.6471 | 0.3354 |
| sqiquery_gm_focus_raw | low_amp_ratio | 0.6723 | 0.3681 |
| sqiquery_gm_focus_raw | detector_agreement | 0.6807 | 0.2520 |
| sqiquery_multiview_robust3 | pca_margin | 0.3128 | 0.7434 |
| sqiquery_multiview_robust3 | pc1 | 0.3848 | 0.7275 |
| sqiquery_multiview_robust3 | non_qrs_diff_p95 | 0.3960 | 0.7080 |
| sqiquery_multiview_robust3 | sqi_bSQI | 0.3978 | 0.6244 |
| sqiquery_multiview_robust3 | boundary_confidence | 0.6216 | 0.3241 |
| sqiquery_multiview_robust3 | template_corr | 0.6276 | 0.4525 |
| sqiquery_multiview_robust3 | low_amp_ratio | 0.6613 | 0.3612 |
| sqiquery_multiview_robust3 | knn_label_purity | 0.6780 | 0.4241 |

## Reference And Interpretation

- 47-feature tabular upper bound on original test: acc `0.963548`, good/medium/bad `0.956/0.973/0.927`.
- Original BUT is report-only. No original row is used for training, validation, threshold selection, or candidate selection.
- If a candidate remains below 0.90 original acc, inspect whether key teacher recovery is weak on `pca_margin`, `boundary_confidence`, `knn_label_purity`, `qrs_visibility`, and bad-stress features before the next model change.

## Artifacts

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_sqi_aware_transformer_metrics.csv`
- Feature recovery CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_sqi_aware_transformer_feature_recovery.csv`
- `sqiquery_gm_focus_raw` best_epoch=9 threshold=0.39 run_dir=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_sqi_aware_transformer\N17043_gm_probe\sqiquery_gm_focus_raw`
- `sqiquery_badstress_hardneg_raw` best_epoch=9 threshold=0.50 run_dir=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_sqi_aware_transformer\N17043_gm_probe\sqiquery_badstress_hardneg_raw`
- `sqiquery_multiview_robust3` best_epoch=8 threshold=0.50 run_dir=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_sqi_aware_transformer\N17043_gm_probe\sqiquery_multiview_robust3`
