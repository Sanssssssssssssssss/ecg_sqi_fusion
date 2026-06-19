# Interpretable Clean PTB/BUT Cross-Dataset Checks

Fixed 10s only. No variable-length inputs. Classifier input is waveform-derived channels; interpretable features supervise training only.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | train_domain_val | 0.972077 | 0.965646 | 0.963145 | 0.972538 | 0.984375 | 15 | 0 | 4 |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | ptb_synthetic_test | 0.981548 | 0.980252 | 0.941423 | 0.997565 | 0.979253 | 28 | 2 | 5 |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_clean_margin_ge_5s_keep_outlier_test | 0.667625 | 0.554861 | 0.651299 | 0.700332 | 0.418006 | 1037 | 709 | 172 |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_original_test_all_10s+ | 0.651764 | 0.539618 | 0.632692 | 0.694080 | 0.364964 | 1275 | 810 | 246 |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_original_all_10s+ | 0.797032 | 0.804920 | 0.850026 | 0.643301 | 0.935289 | 2430 | 3224 | 326 |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_bad_outlier_stress | 0.106164 | 0.063983 | 0.000000 | 0.000000 | 0.106164 | 0 | 0 | 246 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | train_domain_val | 0.833502 | 0.766176 | 0.809636 | 0.983333 | 0.975000 | 162 | 1 | 2 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | ptb_synthetic_test | 0.715530 | 0.488304 | 0.663180 | 0.875812 | 0.000000 | 161 | 152 | 241 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_clean_margin_ge_5s_keep_outlier_test | 0.745549 | 0.682422 | 0.975649 | 0.593454 | 0.379421 | 75 | 1589 | 91 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_original_test_all_10s+ | 0.730211 | 0.639517 | 0.974725 | 0.570041 | 0.289538 | 92 | 1902 | 150 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_original_all_10s+ | 0.871799 | 0.881423 | 0.948073 | 0.718479 | 0.934153 | 885 | 2990 | 205 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier | but_bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 150 |

## Checkpoints

- `ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier` (ptb_synthetic): best_epoch=3, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\interpretable_clean_cross_dataset\ptb_synthetic_margin_ge_5s_keep_outlier_dualview_convtx_hier\ckpt_best.pt`
- `but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier` (but_clean): best_epoch=3, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\interpretable_clean_cross_dataset\but_clean_margin_ge_5s_keep_outlier_dualview_convtx_hier\ckpt_best.pt`

## Interpretation Contract

- `ptb_synthetic_test` measures target-aware PTB synthetic learnability.
- `but_clean_*_test` measures cleaned fixed-10s BUT learnability.
- `but_original_*` buckets remain stress reports, not selection.
- Large asymmetry between within-domain and cross-domain performance is evidence of domain/split/label-policy shift, not proof that variable-length is required.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\interpretable_clean_cross_dataset\interpretable_clean_cross_dataset_margin_ge_5s_keep_outlier_dualview_convtx_hier_metrics.csv`