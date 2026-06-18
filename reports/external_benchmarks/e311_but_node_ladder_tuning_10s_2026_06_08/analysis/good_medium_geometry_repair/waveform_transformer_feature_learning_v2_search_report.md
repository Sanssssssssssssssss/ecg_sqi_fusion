# Transformer-Only Waveform Geometry Learning V2

Inference input is ECG waveform-derived channels only. The 47 SQI/geometry columns are training teacher targets and diagnostics only.

## Main Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m | Key Aux MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sqiquery_conformer_v2 | synthetic_test | 0.990261 | 0.985905 | 0.989540 | 0.991883 | 0.983402 | 1 | 6 | 0 | 4 | 0.7143 |
| sqiquery_conformer_v2_badcal | synthetic_test | 0.990774 | 0.986741 | 0.989540 | 0.991883 | 0.987552 | 1 | 6 | 0 | 3 | nan |
| sqiquery_conformer_v2 | original_test_all_10s+ | 0.690221 | 0.575546 | 0.733242 | 0.669679 | 0.530414 | 463 | 743 | 0 | 12 | nan |
| sqiquery_conformer_v2_badcal | original_test_all_10s+ | 0.661791 | 0.559820 | 0.683516 | 0.648893 | 0.608273 | 452 | 607 | 0 | 9 | nan |
| sqiquery_conformer_v2 | original_all_10s+ | 0.720203 | 0.741513 | 0.590389 | 0.812100 | 0.954021 | 6237 | 1047 | 0 | 60 | nan |
| sqiquery_conformer_v2_badcal | original_all_10s+ | 0.708611 | 0.724385 | 0.574312 | 0.798927 | 0.960076 | 6207 | 887 | 0 | 57 | nan |
| sqiquery_conformer_v2 | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 | nan |
| sqiquery_conformer_v2_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 | nan |
| sqiquery_conformer_v2 | bad_outlier_stress | 0.339041 | 0.168798 | 0.000000 | 0.000000 | 0.339041 | 0 | 0 | 0 | 12 | nan |
| sqiquery_conformer_v2_badcal | bad_outlier_stress | 0.448630 | 0.206462 | 0.000000 | 0.000000 | 0.448630 | 0 | 0 | 0 | 9 | nan |

## Missing-Feature Recovery Gates

| Candidate | Feature | Split | Corr | MAE(z) | Gate |
|---|---|---|---:|---:|---|
| sqiquery_conformer_v2 | baseline_step | synthetic_test | 0.3055 | 0.8290 | need >= 0.55 |
| sqiquery_conformer_v2 | detector_agreement | synthetic_test | 0.2357 | 0.6889 | need >= 0.45 |
| sqiquery_conformer_v2 | flatline_ratio | synthetic_test | 0.5419 | 0.7283 | need >= 0.60 |
| sqiquery_conformer_v2 | pca_margin | synthetic_test | 0.7496 | 0.3138 | pass |
| sqiquery_conformer_v2 | qrs_visibility | synthetic_test | 0.1492 | 0.8854 | need >= 0.45 |

## Weakest Key Teacher Targets

| Candidate | Feature | Corr | MAE(z) |
|---|---|---:|---:|
| sqiquery_conformer_v2 | pc2 | 0.0476 | 1.0002 |
| sqiquery_conformer_v2 | qrs_visibility | 0.1492 | 0.8854 |
| sqiquery_conformer_v2 | sqi_basSQI | 0.1795 | 0.9081 |
| sqiquery_conformer_v2 | detector_agreement | 0.2357 | 0.6889 |
| sqiquery_conformer_v2 | region_confidence | 0.3019 | 0.8035 |
| sqiquery_conformer_v2 | baseline_step | 0.3055 | 0.8290 |
| sqiquery_conformer_v2 | boundary_confidence | 0.3412 | 0.6474 |
| sqiquery_conformer_v2 | low_amp_ratio | 0.3706 | 0.6645 |

## Reference And Interpretation

- 47-feature upper bound original test: acc `0.963548`, good/medium/bad `0.956/0.973/0.927`.
- Original BUT is report-only. No original row is used for training, validation, threshold selection, or candidate selection.
- A candidate below 0.90 original acc is not just an accuracy failure; inspect its failed gates and waveform panels before changing architecture.

## Artifacts

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_feature_learning_v2_search_metrics.csv`
- Feature recovery CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_feature_learning_v2_search_feature_recovery.csv`
- `sqiquery_conformer_v2` best_epoch=6 threshold=0.23 feature_gates=1/5 run_dir=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_feature_learning_v2\N17043_gm_probe\sqiquery_conformer_v2`
