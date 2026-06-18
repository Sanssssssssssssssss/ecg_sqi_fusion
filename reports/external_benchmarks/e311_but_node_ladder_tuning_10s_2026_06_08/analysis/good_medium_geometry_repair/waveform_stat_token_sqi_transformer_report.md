# Stat-Token SQI Transformer

Inference uses waveform-derived enhanced stats plus waveform patch tokens. The 47-column SQI/geometry table is teacher-only.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stattoken_bin_teacher_gm | synthetic_test | 0.993337 | 0.991621 | 0.993724 | 0.995942 | 0.979253 | 3 | 4 | 5 |
| stattoken_bin_teacher_gm_badcal | synthetic_test | 0.991799 | 0.989160 | 0.993724 | 0.993506 | 0.979253 | 3 | 4 | 5 |
| stattoken_bin_teacher_gm | original_test_all_10s+ | 0.853014 | 0.678902 | 0.837637 | 0.928378 | 0.177616 | 591 | 317 | 211 |
| stattoken_bin_teacher_gm_badcal | original_test_all_10s+ | 0.857379 | 0.723152 | 0.837637 | 0.927474 | 0.277372 | 591 | 317 | 170 |
| stattoken_bin_teacher_gm | bad_core_nearboundary | 0.613445 | 0.253472 | 0.000000 | 0.000000 | 0.613445 | 0 | 0 | 46 |
| stattoken_bin_teacher_gm_badcal | bad_core_nearboundary | 0.957983 | 0.326180 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| stattoken_bin_teacher_gm | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 165 |
| stattoken_bin_teacher_gm_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 165 |
| stattoken_bin_teacher_controlbad | synthetic_test | 0.984623 | 0.976904 | 0.991632 | 0.982955 | 0.979253 | 3 | 1 | 5 |
| stattoken_bin_teacher_controlbad_badcal | synthetic_test | 0.984623 | 0.976904 | 0.991632 | 0.982955 | 0.979253 | 3 | 1 | 5 |
| stattoken_bin_teacher_controlbad | original_test_all_10s+ | 0.761236 | 0.574990 | 0.743956 | 0.831902 | 0.153285 | 529 | 301 | 172 |
| stattoken_bin_teacher_controlbad_badcal | original_test_all_10s+ | 0.757107 | 0.575036 | 0.740385 | 0.825576 | 0.167883 | 525 | 301 | 166 |
| stattoken_bin_teacher_controlbad | bad_core_nearboundary | 0.075630 | 0.046875 | 0.000000 | 0.000000 | 0.075630 | 0 | 0 | 110 |
| stattoken_bin_teacher_controlbad_badcal | bad_core_nearboundary | 0.126050 | 0.074627 | 0.000000 | 0.000000 | 0.126050 | 0 | 0 | 104 |
| stattoken_bin_teacher_controlbad | bad_outlier_stress | 0.184932 | 0.104046 | 0.000000 | 0.000000 | 0.184932 | 0 | 0 | 62 |
| stattoken_bin_teacher_controlbad_badcal | bad_outlier_stress | 0.184932 | 0.104046 | 0.000000 | 0.000000 | 0.184932 | 0 | 0 | 62 |

## Weak Key Teacher Features

| Candidate | Feature | MAE(z) | Corr |
|---|---|---:|---:|
| stattoken_bin_teacher_controlbad | pc2 | 1.1866 | -0.1464 |
| stattoken_bin_teacher_controlbad | region_confidence | 0.7735 | -0.1429 |
| stattoken_bin_teacher_controlbad | qrs_visibility | 1.4008 | 0.0638 |
| stattoken_bin_teacher_controlbad | amplitude_entropy | 0.7939 | 0.1925 |
| stattoken_bin_teacher_controlbad | sqi_basSQI | 1.4057 | 0.2317 |
| stattoken_bin_teacher_controlbad | detector_agreement | 1.1722 | 0.2348 |
| stattoken_bin_teacher_gm | pc3 | 1.1157 | -0.4368 |
| stattoken_bin_teacher_gm | low_amp_ratio | 0.8754 | -0.2881 |
| stattoken_bin_teacher_gm | pc2 | 0.9469 | 0.0139 |
| stattoken_bin_teacher_gm | qrs_visibility | 1.2398 | 0.0678 |
| stattoken_bin_teacher_gm | sqi_basSQI | 1.3179 | 0.2254 |
| stattoken_bin_teacher_gm | detector_agreement | 1.0493 | 0.2329 |

## Reference

- 47-feature tabular original-test acc `0.963548`, recalls `0.956/0.973/0.927`.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stat_token_sqi_transformer_metrics.csv`
