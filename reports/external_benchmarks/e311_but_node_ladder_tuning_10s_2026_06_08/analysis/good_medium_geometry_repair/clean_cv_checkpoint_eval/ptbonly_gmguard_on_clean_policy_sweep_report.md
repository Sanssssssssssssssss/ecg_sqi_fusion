# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_0af4de989b083919\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean_core_only_margin2 | clean_val | 1.000000 | 0.666667 | 1.000000 | 1.000000 | 0.000000 | 0 | 0 | 0 |
| clean_core_only_margin2 | clean_test | 0.975000 | 0.633595 | 1.000000 | 0.973361 | 0.000000 | 0 | 6 | 0 |
| clean_core_plus_overlap_margin2 | clean_val | 0.875549 | 0.477265 | 0.869906 | 0.955556 | 0.000000 | 83 | 2 | 0 |
| clean_core_plus_overlap_margin2 | clean_test | 0.925749 | 0.613253 | 0.938521 | 0.919409 | 0.000000 | 64 | 161 | 0 |
| margin_ge_5s_drop_outlier | clean_val | 0.873684 | 0.474203 | 0.869565 | 0.953488 | 0.000000 | 81 | 2 | 1 |
| margin_ge_5s_drop_outlier | clean_test | 0.929040 | 0.934530 | 0.939243 | 0.920077 | 1.000000 | 61 | 157 | 0 |

## Clean Test Summary

- acc: mean=0.943263, min=0.925749, max=0.975000, std=0.022482
- macro_f1: mean=0.727126, min=0.613253, max=0.934530, std=0.146892
- good_recall: mean=0.959255, min=0.938521, max=1.000000, std=0.028813
- medium_recall: mean=0.937615, min=0.919409, max=0.973361, std=0.025277
- bad_recall: mean=0.333333, min=0.000000, max=1.000000, std=0.471405

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\ptbonly_gmguard_on_clean_policy_sweep_metrics.csv`