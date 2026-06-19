# Interpretable Clean PTB/BUT Cross-Dataset Checks

Fixed 10s only. No variable-length inputs. Classifier input is waveform-derived channels; interpretable features supervise training only.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | train_domain_val | 0.963933 | 0.963544 | 0.941032 | 0.969697 | 0.976562 | 24 | 32 | 6 |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | ptb_synthetic_test | 0.968734 | 0.968394 | 0.920502 | 0.985390 | 0.979253 | 38 | 18 | 5 |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_clean_margin_ge_5s_keep_outlier_test | 0.708436 | 0.495444 | 0.687662 | 0.779340 | 0.022508 | 962 | 731 | 291 |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_original_test_all_10s+ | 0.692934 | 0.490917 | 0.669231 | 0.773836 | 0.031630 | 1199 | 853 | 379 |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_original_all_10s+ | 0.794393 | 0.800331 | 0.848266 | 0.706342 | 0.797729 | 2578 | 2968 | 1049 |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_bad_core_nearboundary | 0.016807 | 0.011019 | 0.000000 | 0.000000 | 0.016807 | 0 | 0 | 117 |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_bad_outlier_stress | 0.037671 | 0.024202 | 0.000000 | 0.000000 | 0.037671 | 0 | 0 | 262 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | train_domain_val | 0.994955 | 0.979465 | 1.000000 | 0.966667 | 0.962500 | 0 | 2 | 3 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | ptb_synthetic_test | 0.654024 | 0.458036 | 0.861925 | 0.701299 | 0.000000 | 66 | 368 | 241 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_clean_margin_ge_5s_keep_outlier_test | 0.718707 | 0.662226 | 0.997078 | 0.526464 | 0.379421 | 9 | 1852 | 73 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_original_test_all_10s+ | 0.701663 | 0.617045 | 0.996154 | 0.497741 | 0.289538 | 14 | 2223 | 123 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_original_all_10s+ | 0.843852 | 0.848237 | 0.988852 | 0.567181 | 0.932640 | 190 | 4600 | 186 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 123 |

## Checkpoints

- `ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier` (ptb_synthetic): best_epoch=2, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\interpretable_clean_cross_dataset\ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier\ckpt_best.pt`
- `but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier` (but_clean): best_epoch=3, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\interpretable_clean_cross_dataset\but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier\ckpt_best.pt`

## Interpretation Contract

- `ptb_synthetic_test` measures target-aware PTB synthetic learnability.
- `but_clean_*_test` measures cleaned fixed-10s BUT learnability.
- `but_original_*` buckets remain stress reports, not selection.
- Large asymmetry between within-domain and cross-domain performance is evidence of domain/split/label-policy shift, not proof that variable-length is required.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\interpretable_clean_cross_dataset\interpretable_clean_cross_dataset_metrics.csv`