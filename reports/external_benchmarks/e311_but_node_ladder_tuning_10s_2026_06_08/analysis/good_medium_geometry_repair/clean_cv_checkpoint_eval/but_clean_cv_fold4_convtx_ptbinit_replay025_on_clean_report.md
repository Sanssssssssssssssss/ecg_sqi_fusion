# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold4\dualview_convtx_hier_ptbinit_b0p5_replay025\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_val | 0.981567 | 0.983225 | 0.996900 | 0.943038 | 0.998778 | 7 | 72 | 1 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_test | 0.979026 | 0.980879 | 0.999553 | 0.928226 | 1.000000 | 1 | 89 | 0 |

## Clean Test Summary

- acc: mean=0.979026, min=0.979026, max=0.979026, std=0.000000
- macro_f1: mean=0.980879, min=0.980879, max=0.980879, std=0.000000
- good_recall: mean=0.999553, min=0.999553, max=0.999553, std=0.000000
- medium_recall: mean=0.928226, min=0.928226, max=0.928226, std=0.000000
- bad_recall: mean=1.000000, min=1.000000, max=1.000000, std=0.000000

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\but_clean_cv_fold4_convtx_ptbinit_replay025_on_clean_metrics.csv`