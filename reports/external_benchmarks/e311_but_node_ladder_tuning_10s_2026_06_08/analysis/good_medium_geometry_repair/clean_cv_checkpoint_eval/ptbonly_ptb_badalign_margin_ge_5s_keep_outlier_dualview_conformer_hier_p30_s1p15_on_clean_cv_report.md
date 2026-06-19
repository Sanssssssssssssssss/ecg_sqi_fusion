# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p15\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_val | 0.880721 | 0.881750 | 0.981342 | 0.623211 | 1.000000 | 42 | 468 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_test | 0.881797 | 0.883381 | 0.981399 | 0.627373 | 1.000000 | 42 | 467 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_val | 0.875348 | 0.876182 | 0.976827 | 0.612749 | 1.000000 | 52 | 479 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_test | 0.880721 | 0.881750 | 0.981342 | 0.623211 | 1.000000 | 42 | 468 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_val | 0.878196 | 0.878807 | 0.980348 | 0.615385 | 1.000000 | 44 | 474 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_test | 0.875348 | 0.876182 | 0.976827 | 0.612749 | 1.000000 | 52 | 479 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_val | 0.882545 | 0.883184 | 0.981664 | 0.626613 | 1.000000 | 41 | 457 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_test | 0.878196 | 0.878807 | 0.980348 | 0.615385 | 1.000000 | 44 | 474 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_val | 0.881797 | 0.883381 | 0.981399 | 0.627373 | 1.000000 | 42 | 467 | 0 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_test | 0.882545 | 0.883184 | 0.981664 | 0.626613 | 1.000000 | 41 | 457 | 0 |

## Clean Test Summary

- acc: mean=0.879721, min=0.875348, max=0.882545, std=0.002636
- macro_f1: mean=0.880661, min=0.876182, max=0.883381, std=0.002773
- good_recall: mean=0.980316, min=0.976827, max=0.981664, std=0.001801
- medium_recall: mean=0.621066, min=0.612749, max=0.627373, std=0.005943
- bad_recall: mean=1.000000, min=1.000000, max=1.000000, std=0.000000

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\ptbonly_ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p15_on_clean_cv_metrics.csv`