# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold3\dualview_convtx_hier_ptbinit_b0p5_replay025\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_val | 0.979958 | 0.982199 | 0.968694 | 0.987097 | 1.000000 | 70 | 16 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_test | 0.979312 | 0.981575 | 0.964716 | 0.991987 | 1.000000 | 79 | 9 | 0 |

## Clean Test Summary

- acc: mean=0.979312, min=0.979312, max=0.979312, std=0.000000
- macro_f1: mean=0.981575, min=0.981575, max=0.981575, std=0.000000
- good_recall: mean=0.964716, min=0.964716, max=0.964716, std=0.000000
- medium_recall: mean=0.991987, min=0.991987, max=0.991987, std=0.000000
- bad_recall: mean=1.000000, min=1.000000, max=1.000000, std=0.000000

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\but_clean_cv_fold3_convtx_ptbinit_replay025_on_clean_metrics.csv`