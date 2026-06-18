# Waveform Stat Atlas Proxy Fusion

Input is ECG waveform only. PCA/KNN atlas features are fitted on synthetic train waveform statistics, then recomputed for validation/test/original from waveform statistics.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| statatlas_mlp_balanced | synthetic_test | 0.992312 | 0.991723 | 0.991632 | 0.992695 | 0.991701 | 4 | 8 | 2 |
| statatlas_mlp_balanced_badcal | synthetic_test | 0.992312 | 0.991723 | 0.991632 | 0.992695 | 0.991701 | 4 | 8 | 2 |
| statatlas_mlp_balanced | original_test_all_10s+ | 0.823994 | 0.704331 | 0.838462 | 0.861726 | 0.289538 | 588 | 606 | 69 |
| statatlas_mlp_balanced_badcal | original_test_all_10s+ | 0.823994 | 0.704331 | 0.838462 | 0.861726 | 0.289538 | 588 | 606 | 69 |
| statatlas_mlp_balanced | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| statatlas_mlp_balanced_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| statatlas_mlp_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 69 |
| statatlas_mlp_balanced_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 69 |
| statatlas_mlp_medium | synthetic_test | 0.990774 | 0.990216 | 0.974895 | 0.996753 | 0.991701 | 12 | 3 | 2 |
| statatlas_mlp_medium_badcal | synthetic_test | 0.990774 | 0.990216 | 0.974895 | 0.996753 | 0.991701 | 12 | 3 | 2 |
| statatlas_mlp_medium | original_test_all_10s+ | 0.813849 | 0.696076 | 0.814286 | 0.862404 | 0.287105 | 676 | 603 | 70 |
| statatlas_mlp_medium_badcal | original_test_all_10s+ | 0.813849 | 0.696076 | 0.814286 | 0.862404 | 0.287105 | 676 | 603 | 70 |
| statatlas_mlp_medium | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| statatlas_mlp_medium_badcal | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| statatlas_mlp_medium | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 69 |
| statatlas_mlp_medium_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 69 |
| statatlas_token_balanced | synthetic_test | 0.990261 | 0.989392 | 0.979079 | 0.995942 | 0.983402 | 10 | 5 | 4 |
| statatlas_token_balanced_badcal | synthetic_test | 0.990261 | 0.989402 | 0.979079 | 0.995130 | 0.987552 | 10 | 5 | 3 |
| statatlas_token_balanced | original_test_all_10s+ | 0.825882 | 0.702878 | 0.789835 | 0.905558 | 0.287105 | 765 | 409 | 101 |
| statatlas_token_balanced_badcal | original_test_all_10s+ | 0.824938 | 0.701090 | 0.789835 | 0.903525 | 0.289538 | 765 | 409 | 100 |
| statatlas_token_balanced | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| statatlas_token_balanced_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| statatlas_token_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 100 |
| statatlas_token_balanced_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 100 |
| statatlas_token_badguard | synthetic_test | 0.990774 | 0.990228 | 0.979079 | 0.995942 | 0.987552 | 10 | 5 | 3 |
| statatlas_token_badguard_badcal | synthetic_test | 0.990774 | 0.990228 | 0.979079 | 0.995942 | 0.987552 | 10 | 5 | 3 |
| statatlas_token_badguard | original_test_all_10s+ | 0.826118 | 0.705627 | 0.817582 | 0.883190 | 0.287105 | 664 | 516 | 89 |
| statatlas_token_badguard_badcal | original_test_all_10s+ | 0.826000 | 0.705305 | 0.817582 | 0.882964 | 0.287105 | 664 | 516 | 89 |
| statatlas_token_badguard | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| statatlas_token_badguard_badcal | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| statatlas_token_badguard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 88 |
| statatlas_token_badguard_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 88 |

## Contract

- No original BUT rows are used for atlas fitting, training, validation, threshold selection, or model selection.
- Original BUT is bucketed report-only.
- The atlas is target-aware only through synthetic train labels and waveform statistics.

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stat_atlas_proxy_fusion_metrics.csv`
- Feature schema: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stat_atlas_proxy_feature_schema.json`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stat_atlas_proxy_fusion_summary.json`
