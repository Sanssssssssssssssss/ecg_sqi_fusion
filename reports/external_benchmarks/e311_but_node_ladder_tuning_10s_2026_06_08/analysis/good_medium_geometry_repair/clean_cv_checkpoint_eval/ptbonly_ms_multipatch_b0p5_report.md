# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_drop_outlier_dualview_multipatch_hier_p00_s0p0_distbankbb7b93_b0p5\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_val | 0.928109 | 0.931048 | 0.992892 | 0.771860 | 0.990208 | 16 | 286 | 8 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_test | 0.918664 | 0.922291 | 0.986714 | 0.750000 | 0.991443 | 30 | 316 | 6 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_val | 0.922845 | 0.925754 | 0.990642 | 0.758566 | 0.988984 | 21 | 302 | 9 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_test | 0.928109 | 0.931048 | 0.992892 | 0.771860 | 0.990208 | 16 | 286 | 8 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_val | 0.921432 | 0.924556 | 0.986155 | 0.761218 | 0.988957 | 31 | 297 | 9 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_test | 0.922845 | 0.925754 | 0.990642 | 0.758566 | 0.988984 | 21 | 302 | 9 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_val | 0.922862 | 0.925732 | 0.989267 | 0.757258 | 0.992638 | 24 | 299 | 6 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_test | 0.921432 | 0.924556 | 0.986155 | 0.761218 | 0.988957 | 31 | 297 | 9 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_val | 0.918664 | 0.922291 | 0.986714 | 0.750000 | 0.991443 | 30 | 316 | 6 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_test | 0.922862 | 0.925732 | 0.989267 | 0.757258 | 0.992638 | 24 | 299 | 6 |

## Clean Test Summary

- acc: mean=0.922782, min=0.918664, max=0.928109, std=0.003072
- macro_f1: mean=0.925876, min=0.922291, max=0.931048, std=0.002877
- good_recall: mean=0.989134, min=0.986155, max=0.992892, std=0.002496
- medium_recall: mean=0.759780, min=0.750000, max=0.771860, std=0.007092
- bad_recall: mean=0.990446, min=0.988957, max=0.992638, std=0.001429

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\ptbonly_ms_multipatch_b0p5_metrics.csv`