# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold0\dualview_convtx_hier_ptbinit_b0p5_replay025\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_val | 0.984975 | 0.985897 | 0.988005 | 0.969793 | 1.000000 | 27 | 33 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_test | 0.983871 | 0.984672 | 0.986714 | 0.969146 | 0.998778 | 30 | 33 | 1 |

## Clean Test Summary

- acc: mean=0.983871, min=0.983871, max=0.983871, std=0.000000
- macro_f1: mean=0.984672, min=0.984672, max=0.984672, std=0.000000
- good_recall: mean=0.986714, min=0.986714, max=0.986714, std=0.000000
- medium_recall: mean=0.969146, min=0.969146, max=0.969146, std=0.000000
- bad_recall: mean=0.998778, min=0.998778, max=0.998778, std=0.000000

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\but_clean_cv_fold0_convtx_ptbinit_replay025_on_clean_metrics.csv`