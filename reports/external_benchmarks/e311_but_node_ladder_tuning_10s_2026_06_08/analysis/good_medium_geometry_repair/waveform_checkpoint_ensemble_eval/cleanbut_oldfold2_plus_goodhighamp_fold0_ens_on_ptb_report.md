# Waveform Checkpoint Ensemble Evaluation

Read-only probability averaging of waveform Transformer checkpoints.

## Checkpoints

- `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold2\dualview_convtx_hier_cleanbadaug_c5_s2p2_badstrong\ckpt_best.pt`
- `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold0\dualview_convtx_hier_goodhighamp_c1_bad5\ckpt_best.pt`

## Metrics

| Target | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_val | 0.903432 | 0.889483 | 0.857494 | 0.901515 | 0.984375 | 58 | 11 | 4 |
| ptb_synthetic_test | 0.931317 | 0.925502 | 0.828452 | 0.961851 | 0.979253 | 81 | 30 | 5 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_checkpoint_ensemble_eval\cleanbut_oldfold2_plus_goodhighamp_fold0_ens_on_ptb_metrics.csv`