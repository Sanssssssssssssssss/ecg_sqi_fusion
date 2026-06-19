# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p4\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_val | 0.908229 | 0.911512 | 0.989338 | 0.703498 | 1.000000 | 24 | 372 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_test | 0.902995 | 0.906434 | 0.984942 | 0.694620 | 0.998778 | 34 | 385 | 1 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_val | 0.904773 | 0.908513 | 0.983957 | 0.701195 | 1.000000 | 36 | 375 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_test | 0.908229 | 0.911512 | 0.989338 | 0.703498 | 1.000000 | 24 | 372 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_val | 0.905393 | 0.909259 | 0.980348 | 0.709135 | 1.000000 | 44 | 362 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_test | 0.904773 | 0.908513 | 0.983957 | 0.701195 | 1.000000 | 36 | 375 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_val | 0.903519 | 0.906494 | 0.986583 | 0.690323 | 1.000000 | 30 | 384 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_test | 0.905393 | 0.909259 | 0.980348 | 0.709135 | 1.000000 | 44 | 362 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_val | 0.902995 | 0.906434 | 0.984942 | 0.694620 | 0.998778 | 34 | 385 | 1 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_test | 0.903519 | 0.906494 | 0.986583 | 0.690323 | 1.000000 | 30 | 384 | 0 |

## Clean Test Summary

- acc: mean=0.904982, min=0.902995, max=0.908229, std=0.001835
- macro_f1: mean=0.908443, min=0.906434, max=0.911512, std=0.001893
- good_recall: mean=0.985034, min=0.980348, max=0.989338, std=0.002968
- medium_recall: mean=0.699754, min=0.690323, max=0.709135, std=0.006624
- bad_recall: mean=0.999756, min=0.998778, max=1.000000, std=0.000489

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\ptbonly_ms_b0p4_metrics.csv`