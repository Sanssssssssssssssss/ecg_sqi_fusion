# SQI-Aware Two-Head Bad Veto Evaluation

Evaluation-only: each threshold is selected on synthetic validation, then BUT is report-only.

## Metrics

| Candidate | Threshold | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.990 | synthetic_test | 0.984111 | 0.974866 | 0.991632 | 0.982143 | 0.979253 | 1 | 0 | 5 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.990 | original_test_all_10s+ | 0.290905 | 0.294810 | 0.204396 | 0.319476 | 0.749392 | 313 | 75 | 64 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.990 | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.990 | bad_outlier_stress | 0.647260 | 0.261954 | 0.000000 | 0.000000 | 0.647260 | 0 | 0 | 64 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.995 | synthetic_test | 0.984111 | 0.974866 | 0.991632 | 0.982143 | 0.979253 | 1 | 0 | 5 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.995 | original_test_all_10s+ | 0.290905 | 0.294810 | 0.204396 | 0.319476 | 0.749392 | 313 | 75 | 64 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.995 | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.995 | bad_outlier_stress | 0.647260 | 0.261954 | 0.000000 | 0.000000 | 0.647260 | 0 | 0 | 64 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.999 | synthetic_test | 0.990774 | 0.985194 | 0.993724 | 0.991883 | 0.979253 | 1 | 0 | 5 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.999 | original_test_all_10s+ | 0.375015 | 0.358775 | 0.264286 | 0.438997 | 0.666667 | 377 | 90 | 87 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.999 | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.999 | bad_outlier_stress | 0.530822 | 0.231171 | 0.000000 | 0.000000 | 0.530822 | 0 | 0 | 87 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_1.000 | synthetic_test | 0.990774 | 0.985194 | 0.993724 | 0.991883 | 0.979253 | 1 | 0 | 5 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_1.000 | original_test_all_10s+ | 0.375015 | 0.358775 | 0.264286 | 0.438997 | 0.666667 | 377 | 90 | 87 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_1.000 | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_1.000 | bad_outlier_stress | 0.530822 | 0.231171 | 0.000000 | 0.000000 | 0.530822 | 0 | 0 | 87 |
| sqiquery_gm_focus_raw | nonbad_floor_0.990 | synthetic_test | 0.993337 | 0.992299 | 0.983264 | 0.999188 | 0.983402 | 8 | 1 | 4 |
| sqiquery_gm_focus_raw | nonbad_floor_0.990 | original_test_all_10s+ | 0.793323 | 0.616494 | 0.667582 | 0.957298 | 0.141119 | 1210 | 188 | 290 |
| sqiquery_gm_focus_raw | nonbad_floor_0.990 | bad_core_nearboundary | 0.487395 | 0.218456 | 0.000000 | 0.000000 | 0.487395 | 0 | 0 | 61 |
| sqiquery_gm_focus_raw | nonbad_floor_0.990 | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 229 |
| sqiquery_gm_focus_raw | nonbad_floor_0.995 | synthetic_test | 0.993337 | 0.992299 | 0.983264 | 0.999188 | 0.983402 | 8 | 1 | 4 |
| sqiquery_gm_focus_raw | nonbad_floor_0.995 | original_test_all_10s+ | 0.793323 | 0.616494 | 0.667582 | 0.957298 | 0.141119 | 1210 | 188 | 290 |
| sqiquery_gm_focus_raw | nonbad_floor_0.995 | bad_core_nearboundary | 0.487395 | 0.218456 | 0.000000 | 0.000000 | 0.487395 | 0 | 0 | 61 |
| sqiquery_gm_focus_raw | nonbad_floor_0.995 | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 229 |
| sqiquery_gm_focus_raw | nonbad_floor_0.999 | synthetic_test | 0.993337 | 0.992299 | 0.983264 | 0.999188 | 0.983402 | 8 | 1 | 4 |
| sqiquery_gm_focus_raw | nonbad_floor_0.999 | original_test_all_10s+ | 0.793323 | 0.616494 | 0.667582 | 0.957298 | 0.141119 | 1210 | 188 | 290 |
| sqiquery_gm_focus_raw | nonbad_floor_0.999 | bad_core_nearboundary | 0.487395 | 0.218456 | 0.000000 | 0.000000 | 0.487395 | 0 | 0 | 61 |
| sqiquery_gm_focus_raw | nonbad_floor_0.999 | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 229 |
| sqiquery_gm_focus_raw | nonbad_floor_1.000 | synthetic_test | 0.993337 | 0.992299 | 0.983264 | 0.999188 | 0.983402 | 8 | 1 | 4 |
| sqiquery_gm_focus_raw | nonbad_floor_1.000 | original_test_all_10s+ | 0.793323 | 0.616494 | 0.667582 | 0.957298 | 0.141119 | 1210 | 188 | 290 |
| sqiquery_gm_focus_raw | nonbad_floor_1.000 | bad_core_nearboundary | 0.487395 | 0.218456 | 0.000000 | 0.000000 | 0.487395 | 0 | 0 | 61 |
| sqiquery_gm_focus_raw | nonbad_floor_1.000 | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 229 |
| sqiquery_multiview_robust3 | nonbad_floor_0.990 | synthetic_test | 0.992824 | 0.991465 | 0.985356 | 0.998377 | 0.979253 | 7 | 2 | 5 |
| sqiquery_multiview_robust3 | nonbad_floor_0.990 | original_test_all_10s+ | 0.749204 | 0.522237 | 0.810989 | 0.766380 | 0.017032 | 679 | 1028 | 154 |
| sqiquery_multiview_robust3 | nonbad_floor_0.990 | bad_core_nearboundary | 0.033613 | 0.021680 | 0.000000 | 0.000000 | 0.033613 | 0 | 0 | 115 |
| sqiquery_multiview_robust3 | nonbad_floor_0.990 | bad_outlier_stress | 0.010274 | 0.006780 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 39 |
| sqiquery_multiview_robust3 | nonbad_floor_0.995 | synthetic_test | 0.992824 | 0.991465 | 0.985356 | 0.998377 | 0.979253 | 7 | 2 | 5 |
| sqiquery_multiview_robust3 | nonbad_floor_0.995 | original_test_all_10s+ | 0.749204 | 0.522237 | 0.810989 | 0.766380 | 0.017032 | 679 | 1028 | 154 |
| sqiquery_multiview_robust3 | nonbad_floor_0.995 | bad_core_nearboundary | 0.033613 | 0.021680 | 0.000000 | 0.000000 | 0.033613 | 0 | 0 | 115 |
| sqiquery_multiview_robust3 | nonbad_floor_0.995 | bad_outlier_stress | 0.010274 | 0.006780 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 39 |
| sqiquery_multiview_robust3 | nonbad_floor_0.999 | synthetic_test | 0.992824 | 0.991465 | 0.985356 | 0.998377 | 0.979253 | 7 | 2 | 5 |
| sqiquery_multiview_robust3 | nonbad_floor_0.999 | original_test_all_10s+ | 0.749204 | 0.522237 | 0.810989 | 0.766380 | 0.017032 | 679 | 1028 | 154 |
| sqiquery_multiview_robust3 | nonbad_floor_0.999 | bad_core_nearboundary | 0.033613 | 0.021680 | 0.000000 | 0.000000 | 0.033613 | 0 | 0 | 115 |
| sqiquery_multiview_robust3 | nonbad_floor_0.999 | bad_outlier_stress | 0.010274 | 0.006780 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 39 |
| sqiquery_multiview_robust3 | nonbad_floor_1.000 | synthetic_test | 0.992824 | 0.991465 | 0.985356 | 0.998377 | 0.979253 | 7 | 2 | 5 |
| sqiquery_multiview_robust3 | nonbad_floor_1.000 | original_test_all_10s+ | 0.749204 | 0.522237 | 0.810989 | 0.766380 | 0.017032 | 679 | 1028 | 154 |
| sqiquery_multiview_robust3 | nonbad_floor_1.000 | bad_core_nearboundary | 0.033613 | 0.021680 | 0.000000 | 0.000000 | 0.033613 | 0 | 0 | 115 |
| sqiquery_multiview_robust3 | nonbad_floor_1.000 | bad_outlier_stress | 0.010274 | 0.006780 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 39 |

## Synthetic-Val Thresholds

| Candidate | Threshold | Value | Val nonbad exact | Val bad R | Val acc |
|---|---|---:|---:|---:|---:|
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.990 | 0.470 | 0.997949 | 0.980469 | 0.995346 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.995 | 0.470 | 0.997949 | 0.980469 | 0.995346 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_0.999 | 0.640 | 0.998633 | 0.976562 | 0.995346 |
| sqiquery_badstress_hardneg_raw | nonbad_floor_1.000 | 0.640 | 0.998633 | 0.976562 | 0.995346 |
| sqiquery_gm_focus_raw | nonbad_floor_0.990 | 0.545 | 1.000000 | 0.988281 | 0.998255 |
| sqiquery_gm_focus_raw | nonbad_floor_0.995 | 0.545 | 1.000000 | 0.988281 | 0.998255 |
| sqiquery_gm_focus_raw | nonbad_floor_0.999 | 0.545 | 1.000000 | 0.988281 | 0.998255 |
| sqiquery_gm_focus_raw | nonbad_floor_1.000 | 0.545 | 1.000000 | 0.988281 | 0.998255 |
| sqiquery_multiview_robust3 | nonbad_floor_0.990 | 0.990 | 0.999316 | 0.976562 | 0.995928 |
| sqiquery_multiview_robust3 | nonbad_floor_0.995 | 0.990 | 0.999316 | 0.976562 | 0.995928 |
| sqiquery_multiview_robust3 | nonbad_floor_0.999 | 0.990 | 0.999316 | 0.976562 | 0.995928 |
| sqiquery_multiview_robust3 | nonbad_floor_1.000 | 0.990 | 0.999316 | 0.976562 | 0.995928 |

## Readout

- If two-head improves original bad without destroying good/medium, the encoder already has a usable bad signal and we should train this form directly.
- If it behaves like the softmax variants, the bad stress signal is not aligned enough and the next move should be better bad/nonbad stress generation plus stronger teacher recovery.
