# Waveform Checkpoint Ensemble Evaluation

Read-only probability averaging of waveform Transformer checkpoints.

## Checkpoints

- `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_0af4de989b083919\ckpt_best.pt`
- `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_6235d1a37fd05b3c\ckpt_best.pt`

## Metrics

| Target | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_val | 0.935338 | 0.538527 | 0.937198 | 0.930233 | 0.000000 | 39 | 3 | 1 |
| margin_ge_5s_drop_outlier_test | 0.912160 | 0.932549 | 0.957171 | 0.885412 | 1.000000 | 43 | 237 | 0 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_checkpoint_ensemble_eval\ptbonly_oldbest_plus_gmguard_ens_on_clean_margin5_metrics.csv`