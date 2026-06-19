# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold2\dualview_convtx_hier_ptbinit_b0p5_replay025\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_val | 0.974430 | 0.976600 | 0.995534 | 0.919872 | 1.000000 | 10 | 99 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_test | 0.974282 | 0.976340 | 0.996435 | 0.917928 | 1.000000 | 8 | 101 | 0 |

## Clean Test Summary

- acc: mean=0.974282, min=0.974282, max=0.974282, std=0.000000
- macro_f1: mean=0.976340, min=0.976340, max=0.976340, std=0.000000
- good_recall: mean=0.996435, min=0.996435, max=0.996435, std=0.000000
- medium_recall: mean=0.917928, min=0.917928, max=0.917928, std=0.000000
- bad_recall: mean=1.000000, min=1.000000, max=1.000000, std=0.000000

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\but_clean_cv_fold2_convtx_ptbinit_replay025_on_clean_metrics.csv`