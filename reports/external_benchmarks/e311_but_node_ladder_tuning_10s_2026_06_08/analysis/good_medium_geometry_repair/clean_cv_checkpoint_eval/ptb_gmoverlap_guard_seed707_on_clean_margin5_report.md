# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_6235d1a37fd05b3c\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier | clean_val | 0.975940 | 0.603372 | 0.985507 | 0.860465 | 0.000000 | 9 | 6 | 1 |
| margin_ge_5s_drop_outlier | clean_test | 0.779931 | 0.655203 | 0.994024 | 0.706789 | 0.245763 | 6 | 609 | 89 |

## Clean Test Summary

- acc: mean=0.779931, min=0.779931, max=0.779931, std=0.000000
- macro_f1: mean=0.655203, min=0.655203, max=0.655203, std=0.000000
- good_recall: mean=0.994024, min=0.994024, max=0.994024, std=0.000000
- medium_recall: mean=0.706789, min=0.706789, max=0.706789, std=0.000000
- bad_recall: mean=0.245763, min=0.245763, max=0.245763, std=0.000000

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\ptb_gmoverlap_guard_seed707_on_clean_margin5_metrics.csv`