# Waveform Checkpoint Ensemble Evaluation

Read-only probability averaging of waveform Transformer checkpoints.

## Checkpoints

- `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_0af4de989b083919\ckpt_best.pt`
- `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_drop_outlier_dualview_multipatch_hier_p00_s0p0_distbankbb7b93_b0p5\ckpt_best.pt`

## Metrics

| Target | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| margin_ge_5s_drop_outlier_val | 0.912782 | 0.510617 | 0.913043 | 0.930233 | 0.000000 | 54 | 3 | 1 |
| margin_ge_5s_drop_outlier_test | 0.938418 | 0.947638 | 0.950199 | 0.929225 | 1.000000 | 50 | 143 | 0 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_checkpoint_ensemble_eval\ptbonly_oldbest055_multipatch045_weighted_on_clean_margin5_metrics.csv`