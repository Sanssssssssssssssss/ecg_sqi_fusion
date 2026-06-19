# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p55\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_val | 0.804207 | 0.778631 | 0.998223 | 0.329889 | 1.000000 | 4 | 842 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_test | 0.796083 | 0.766532 | 0.998229 | 0.303797 | 0.998778 | 4 | 879 | 1 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_val | 0.793327 | 0.763206 | 0.995098 | 0.298008 | 1.000000 | 11 | 881 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_test | 0.804207 | 0.778631 | 0.998223 | 0.329889 | 1.000000 | 4 | 842 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_val | 0.796839 | 0.766951 | 0.997767 | 0.303686 | 1.000000 | 5 | 869 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_test | 0.793327 | 0.763206 | 0.995098 | 0.298008 | 1.000000 | 11 | 881 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_val | 0.796784 | 0.766263 | 0.996869 | 0.302419 | 1.000000 | 7 | 865 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_test | 0.796839 | 0.766951 | 0.997767 | 0.303686 | 1.000000 | 5 | 869 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_val | 0.796083 | 0.766532 | 0.998229 | 0.303797 | 0.998778 | 4 | 879 | 1 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_test | 0.796784 | 0.766263 | 0.996869 | 0.302419 | 1.000000 | 7 | 865 | 0 |

## Clean Test Summary

- acc: mean=0.797448, min=0.793327, max=0.804207, std=0.003615
- macro_f1: mean=0.768317, min=0.763206, max=0.778631, std=0.005325
- good_recall: mean=0.997237, min=0.995098, max=0.998229, std=0.001179
- medium_recall: mean=0.307560, min=0.298008, max=0.329889, std=0.011361
- bad_recall: mean=0.999756, min=0.998778, max=1.000000, std=0.000489

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\ptbonly_ms_b0p55_metrics.csv`