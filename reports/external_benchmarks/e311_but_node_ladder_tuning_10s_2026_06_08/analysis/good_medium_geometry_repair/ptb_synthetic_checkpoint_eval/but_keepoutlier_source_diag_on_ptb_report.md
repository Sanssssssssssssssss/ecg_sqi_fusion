# Checkpoint On PTB Synthetic

Read-only evaluation of an existing waveform-only checkpoint on PTB synthetic val/test splits.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_keep_outlier\dualview_convtx_hier_keepoutlier_source_diag\ckpt_best.pt`

| Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_val | 0.703316 | 0.501733 | 0.862408 | 0.812500 | 0.000000 | 56 | 198 | 256 |
| ptb_synthetic_test | 0.658124 | 0.455495 | 0.769874 | 0.743506 | 0.000000 | 110 | 316 | 241 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_synthetic_checkpoint_eval\but_keepoutlier_source_diag_on_ptb_metrics.csv`