# Clean-BUT CV Checkpoint Evaluation

This report evaluates an existing waveform-only checkpoint on clean fixed-10s CV folds.
No training is performed and no legacy full/original BUT buckets are used here.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p75\ckpt_best.pt`

## Metrics

| Policy | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_val | 0.832178 | 0.779198 | 0.953354 | 0.833863 | 0.495716 | 105 | 209 | 412 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold0 | clean_test | 0.825346 | 0.771326 | 0.950841 | 0.820411 | 0.486553 | 111 | 227 | 420 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_val | 0.825070 | 0.769814 | 0.950980 | 0.824701 | 0.479804 | 110 | 220 | 425 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold1 | clean_test | 0.832178 | 0.779198 | 0.953354 | 0.833863 | 0.495716 | 105 | 209 | 412 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_val | 0.832171 | 0.784449 | 0.949531 | 0.824519 | 0.521472 | 113 | 219 | 390 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold2 | clean_test | 0.825070 | 0.769814 | 0.950980 | 0.824701 | 0.479804 | 110 | 220 | 425 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_val | 0.827313 | 0.766913 | 0.959750 | 0.829032 | 0.461350 | 90 | 212 | 439 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold3 | clean_test | 0.832171 | 0.784449 | 0.949531 | 0.824519 | 0.521472 | 113 | 219 | 390 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_val | 0.825346 | 0.771326 | 0.950841 | 0.820411 | 0.486553 | 111 | 227 | 420 |
| margin_ge_5s_drop_outlier_cv_seed20260619_fold4 | clean_test | 0.827313 | 0.766913 | 0.959750 | 0.829032 | 0.461350 | 90 | 212 | 439 |

## Clean Test Summary

- acc: mean=0.828415, min=0.825070, max=0.832178, std=0.003165
- macro_f1: mean=0.774340, min=0.766913, max=0.784449, std=0.006489
- good_recall: mean=0.952891, min=0.949531, max=0.959750, std=0.003644
- medium_recall: mean=0.826505, min=0.820411, max=0.833863, std=0.004580
- bad_recall: mean=0.488979, min=0.461350, max=0.521472, std=0.019772

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_cv_checkpoint_eval\ptbonly_ms_b0p75_metrics.csv`