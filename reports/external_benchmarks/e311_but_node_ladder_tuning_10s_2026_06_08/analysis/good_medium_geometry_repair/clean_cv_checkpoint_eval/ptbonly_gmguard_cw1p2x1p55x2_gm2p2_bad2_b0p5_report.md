# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_0af4de989b083919\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_val | 0.919094 | 0.927977 | 0.915149 | 0.873609 | 1.000000 | 191 | 158 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_test | 0.910599 | 0.920026 | 0.910983 | 0.852848 | 0.998778 | 201 | 184 | 1 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_val | 0.917053 | 0.926175 | 0.914884 | 0.866932 | 1.000000 | 191 | 167 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_test | 0.919094 | 0.927977 | 0.915149 | 0.873609 | 1.000000 | 191 | 158 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_val | 0.917713 | 0.925827 | 0.918714 | 0.862179 | 1.000000 | 182 | 167 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_test | 0.917053 | 0.926175 | 0.914884 | 0.866932 | 1.000000 | 191 | 167 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_val | 0.921697 | 0.929709 | 0.923971 | 0.866129 | 1.000000 | 170 | 165 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_test | 0.917713 | 0.925827 | 0.918714 | 0.862179 | 1.000000 | 182 | 167 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_val | 0.910599 | 0.920026 | 0.910983 | 0.852848 | 0.998778 | 201 | 184 | 1 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_test | 0.921697 | 0.929709 | 0.923971 | 0.866129 | 1.000000 | 170 | 165 | 0 |

## Clean Test Summary

- acc: mean=0.917231, min=0.910599, max=0.921697, std=0.003679
- macro_f1: mean=0.925943, min=0.920026, max=0.929709, std=0.003268
- good_recall: mean=0.916740, min=0.910983, max=0.923971, std=0.004366
- medium_recall: mean=0.864340, min=0.852848, max=0.873609, std=0.006820
- bad_recall: mean=0.999756, min=0.998778, max=1.000000, std=0.000489

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\ptbonly_gmguard_cw1p2x1p55x2_gm2p2_bad2_b0p5_metrics.csv`