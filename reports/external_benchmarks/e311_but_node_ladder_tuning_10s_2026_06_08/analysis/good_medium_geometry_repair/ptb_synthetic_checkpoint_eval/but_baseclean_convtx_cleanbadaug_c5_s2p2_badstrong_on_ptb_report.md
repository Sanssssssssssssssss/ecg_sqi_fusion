# Checkpoint On PTB Synthetic

Read-only evaluation of an existing waveform-only checkpoint on PTB synthetic val/test splits.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier\dualview_convtx_hier_baseclean_cleanbadaug_c5_s2p2_badstrong\ckpt_best.pt`

| Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_val | 0.840023 | 0.824403 | 0.744472 | 0.841856 | 0.984375 | 104 | 62 | 4 |
| ptb_synthetic_test | 0.885187 | 0.870030 | 0.675732 | 0.948052 | 0.979253 | 152 | 31 | 5 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_synthetic_checkpoint_eval\but_baseclean_convtx_cleanbadaug_c5_s2p2_badstrong_on_ptb_metrics.csv`