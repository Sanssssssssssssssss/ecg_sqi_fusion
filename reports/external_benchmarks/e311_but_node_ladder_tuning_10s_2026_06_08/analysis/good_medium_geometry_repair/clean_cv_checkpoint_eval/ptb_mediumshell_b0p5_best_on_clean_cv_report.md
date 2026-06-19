# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_val | 0.917013 | 0.927793 | 0.883163 | 0.923688 | 1.000000 | 263 | 96 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_test | 0.910599 | 0.922074 | 0.879097 | 0.909810 | 0.998778 | 273 | 114 | 1 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_val | 0.919138 | 0.929580 | 0.885918 | 0.925896 | 1.000000 | 256 | 93 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_test | 0.917013 | 0.927793 | 0.883163 | 0.923688 | 1.000000 | 263 | 96 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_val | 0.916783 | 0.927476 | 0.883430 | 0.922276 | 1.000000 | 261 | 97 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_test | 0.919138 | 0.929580 | 0.885918 | 0.925896 | 1.000000 | 256 | 93 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_val | 0.914472 | 0.925454 | 0.878801 | 0.922581 | 1.000000 | 271 | 96 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_test | 0.916783 | 0.927476 | 0.883430 | 0.922276 | 1.000000 | 261 | 97 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_val | 0.910599 | 0.922074 | 0.879097 | 0.909810 | 0.998778 | 273 | 114 | 1 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_test | 0.914472 | 0.925454 | 0.878801 | 0.922581 | 1.000000 | 271 | 96 | 0 |

## Clean Test Summary

- acc: mean=0.915601, min=0.910599, max=0.919138, std=0.002905
- macro_f1: mean=0.926475, min=0.922074, max=0.929580, std=0.002561
- good_recall: mean=0.882082, min=0.878801, max=0.885918, std=0.002734
- medium_recall: mean=0.920850, min=0.909810, max=0.925896, std=0.005664
- bad_recall: mean=0.999756, min=0.998778, max=1.000000, std=0.000489

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\ptb_mediumshell_b0p5_best_on_clean_cv_metrics.csv`