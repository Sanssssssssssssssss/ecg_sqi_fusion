# Checkpoint On PTB Synthetic

Read-only evaluation of an existing waveform-only checkpoint on PTB synthetic val/test splits.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_source80_seed20260619\dualview_convtx_hier_source80_cleanbadaug_c5_s2p2_badstrong\ckpt_best.pt`

| Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_val | 0.739383 | 0.595518 | 0.837838 | 0.856061 | 0.101562 | 66 | 57 | 230 |
| ptb_synthetic_test | 0.791389 | 0.619874 | 0.811715 | 0.915584 | 0.116183 | 87 | 72 | 213 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_synthetic_checkpoint_eval\but_source80_convtx_cleanbadaug_c5_s2p2_badstrong_on_ptb_metrics.csv`