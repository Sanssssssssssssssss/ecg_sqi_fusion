# Waveform Checkpoint Ensemble Evaluation

Read-only probability averaging of waveform Transformer checkpoints.

## Checkpoints

- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold0\dualview_convtx_hier_cleanbadaug_c5_s2p2_badstrong\ckpt_best.pt`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold0\dualview_convtx_hier_cleanbadaug_c5_s2p2_goodtilt\ckpt_best.pt`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold0\dualview_convtx_hier_clean_gmptbaug_g1m1_b5s2p2_badmax\ckpt_best.pt`

## Metrics

| Target | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_val | 0.853403 | 0.842508 | 0.835381 | 0.828598 | 0.984375 | 67 | 89 | 4 |
| ptb_synthetic_test | 0.876474 | 0.876624 | 0.780335 | 0.893669 | 0.979253 | 105 | 114 | 5 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_checkpoint_ensemble_eval\cleanbut_bad0_goodtilt_badmax_ens_on_ptb_metrics.csv`