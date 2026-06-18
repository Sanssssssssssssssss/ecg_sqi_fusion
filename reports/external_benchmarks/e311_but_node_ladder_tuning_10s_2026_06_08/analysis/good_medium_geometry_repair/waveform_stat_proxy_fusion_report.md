# Waveform Stat Proxy Fusion

Model input is deterministic statistics computed from ECG waveform at inference time. No 47-column SQI/geometry sidecar is used as input.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| statproxy_mlp_balanced | synthetic_test | 0.993337 | 0.992314 | 0.989540 | 0.996753 | 0.983402 | 5 | 4 | 4 |
| statproxy_mlp_balanced_badcal | synthetic_test | 0.993337 | 0.992314 | 0.989540 | 0.996753 | 0.983402 | 5 | 4 | 4 |
| statproxy_mlp_balanced | original_test_all_10s+ | 0.832252 | 0.705052 | 0.861813 | 0.858563 | 0.287105 | 503 | 601 | 71 |
| statproxy_mlp_balanced_badcal | original_test_all_10s+ | 0.831780 | 0.704538 | 0.861813 | 0.857433 | 0.289538 | 503 | 601 | 70 |
| statproxy_mlp_balanced | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| statproxy_mlp_balanced_badcal | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| statproxy_mlp_balanced | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 69 |
| statproxy_mlp_balanced_badcal | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 69 |
| statproxy_mlp_medium | synthetic_test | 0.986161 | 0.986360 | 0.989540 | 0.982955 | 0.995851 | 5 | 20 | 1 |
| statproxy_mlp_medium_badcal | synthetic_test | 0.986161 | 0.986360 | 0.989540 | 0.982955 | 0.995851 | 5 | 20 | 1 |
| statproxy_mlp_medium | original_test_all_10s+ | 0.839094 | 0.668943 | 0.868407 | 0.876186 | 0.180049 | 479 | 538 | 119 |
| statproxy_mlp_medium_badcal | original_test_all_10s+ | 0.839094 | 0.668943 | 0.868407 | 0.876186 | 0.180049 | 479 | 538 | 119 |
| statproxy_mlp_medium | bad_core_nearboundary | 0.621849 | 0.255613 | 0.000000 | 0.000000 | 0.621849 | 0 | 0 | 45 |
| statproxy_mlp_medium_badcal | bad_core_nearboundary | 0.621849 | 0.255613 | 0.000000 | 0.000000 | 0.621849 | 0 | 0 | 45 |
| statproxy_mlp_medium | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 74 |
| statproxy_mlp_medium_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 74 |
| statproxy_token_balanced | synthetic_test | 0.987699 | 0.986984 | 0.976987 | 0.992695 | 0.983402 | 11 | 9 | 4 |
| statproxy_token_balanced_badcal | synthetic_test | 0.987699 | 0.986984 | 0.976987 | 0.992695 | 0.983402 | 11 | 9 | 4 |
| statproxy_token_balanced | original_test_all_10s+ | 0.830718 | 0.711939 | 0.764560 | 0.932897 | 0.316302 | 843 | 284 | 121 |
| statproxy_token_balanced_badcal | original_test_all_10s+ | 0.830482 | 0.711713 | 0.764560 | 0.932219 | 0.318735 | 842 | 284 | 120 |
| statproxy_token_balanced | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| statproxy_token_balanced_badcal | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| statproxy_token_balanced | bad_outlier_stress | 0.041096 | 0.026316 | 0.000000 | 0.000000 | 0.041096 | 0 | 0 | 120 |
| statproxy_token_balanced_badcal | bad_outlier_stress | 0.044521 | 0.028415 | 0.000000 | 0.000000 | 0.044521 | 0 | 0 | 119 |
| statproxy_token_badguard | synthetic_test | 0.991799 | 0.991214 | 0.985356 | 0.995130 | 0.987552 | 7 | 6 | 3 |
| statproxy_token_badguard_badcal | synthetic_test | 0.991799 | 0.991214 | 0.985356 | 0.995130 | 0.987552 | 7 | 6 | 3 |
| statproxy_token_badguard | original_test_all_10s+ | 0.723133 | 0.634016 | 0.808242 | 0.693629 | 0.287105 | 698 | 1345 | 37 |
| statproxy_token_badguard_badcal | original_test_all_10s+ | 0.722661 | 0.633413 | 0.808242 | 0.692499 | 0.289538 | 698 | 1345 | 36 |
| statproxy_token_badguard | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| statproxy_token_badguard_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| statproxy_token_badguard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 36 |
| statproxy_token_badguard_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 36 |

## Notes

- Selection uses synthetic train/val only.
- Original BUT is bucketed report-only.
- This is waveform-only at inference because all statistics are recomputed from the input ECG.

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stat_proxy_fusion_metrics.csv`
- Feature schema: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stat_proxy_feature_schema.json`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stat_proxy_fusion_summary.json`
