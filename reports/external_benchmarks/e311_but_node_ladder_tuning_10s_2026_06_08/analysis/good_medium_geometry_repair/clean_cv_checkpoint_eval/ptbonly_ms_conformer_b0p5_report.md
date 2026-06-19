# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_drop_outlier_dualview_conformer_hier_p00_s0p0_distbankbb7b93_b0p5\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_val | 0.630837 | 0.410079 | 0.996446 | 0.386328 | 0.000000 | 8 | 772 | 817 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_test | 0.623272 | 0.401599 | 0.995571 | 0.361551 | 0.000000 | 10 | 807 | 818 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_val | 0.617933 | 0.397159 | 0.991533 | 0.352191 | 0.000000 | 19 | 813 | 817 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_test | 0.630837 | 0.410079 | 0.996446 | 0.386328 | 0.000000 | 8 | 772 | 817 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_val | 0.619014 | 0.397204 | 0.994194 | 0.350160 | 0.000000 | 13 | 811 | 815 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_test | 0.617933 | 0.397159 | 0.991533 | 0.352191 | 0.000000 | 19 | 813 | 817 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_val | 0.618737 | 0.396575 | 0.994186 | 0.348387 | 0.000000 | 13 | 808 | 815 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_test | 0.619014 | 0.397204 | 0.994194 | 0.350160 | 0.000000 | 13 | 811 | 815 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_val | 0.623272 | 0.401599 | 0.995571 | 0.361551 | 0.000000 | 10 | 807 | 818 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_test | 0.618737 | 0.396575 | 0.994186 | 0.348387 | 0.000000 | 13 | 808 | 815 |

## Clean Test Summary

- acc: mean=0.621959, min=0.617933, max=0.630837, std=0.004812
- macro_f1: mean=0.400523, min=0.396575, max=0.410079, std=0.005107
- good_recall: mean=0.994386, min=0.991533, max=0.996446, std=0.001665
- medium_recall: mean=0.359723, min=0.348387, max=0.386328, std=0.014056
- bad_recall: mean=0.000000, min=0.000000, max=0.000000, std=0.000000

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\ptbonly_ms_conformer_b0p5_metrics.csv`