# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold1\dualview_convtx_hier_ptbinit_b0p5_replay025\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_val | 0.982854 | 0.984423 | 0.993316 | 0.952988 | 1.000000 | 15 | 58 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_test | 0.983819 | 0.985313 | 0.992892 | 0.957075 | 1.000000 | 16 | 53 | 0 |

## Clean Test Summary

- acc: mean=0.983819, min=0.983819, max=0.983819, std=0.000000
- macro_f1: mean=0.985313, min=0.985313, max=0.985313, std=0.000000
- good_recall: mean=0.992892, min=0.992892, max=0.992892, std=0.000000
- medium_recall: mean=0.957075, min=0.957075, max=0.957075, std=0.000000
- bad_recall: mean=1.000000, min=1.000000, max=1.000000, std=0.000000

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\but_clean_cv_fold1_convtx_ptbinit_replay025_on_clean_metrics.csv`