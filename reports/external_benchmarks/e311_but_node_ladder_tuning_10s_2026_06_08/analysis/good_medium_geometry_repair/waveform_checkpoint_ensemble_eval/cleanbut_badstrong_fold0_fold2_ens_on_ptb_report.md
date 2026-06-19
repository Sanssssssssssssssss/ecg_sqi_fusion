# Waveform Checkpoint Ensemble Evaluation

Read-only probability averaging of waveform Transformer checkpoints.

## Checkpoints

- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold0\dualview_convtx_hier_cleanbadaug_c5_s2p2_badstrong\ckpt_best.pt`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold2\dualview_convtx_hier_cleanbadaug_c5_s2p2_badstrong\ckpt_best.pt`

## Metrics

| Target | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_val | 0.881326 | 0.869914 | 0.862408 | 0.863636 | 0.984375 | 56 | 62 | 4 |
| ptb_synthetic_test | 0.916966 | 0.913364 | 0.830544 | 0.938312 | 0.979253 | 81 | 59 | 5 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_checkpoint_ensemble_eval\cleanbut_badstrong_fold0_fold2_ens_on_ptb_metrics.csv`