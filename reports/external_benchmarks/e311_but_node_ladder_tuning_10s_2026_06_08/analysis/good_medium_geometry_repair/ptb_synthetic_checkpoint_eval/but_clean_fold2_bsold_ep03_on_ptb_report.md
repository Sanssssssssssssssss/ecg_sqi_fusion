# Checkpoint On PTB Synthetic

Read-only evaluation of an existing waveform-only checkpoint on PTB synthetic val/test splits.

Checkpoint: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold2\dualview_convtx_hier_bsold_ep\ckpt_epoch03.pt`

| Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_test | 0.875961 | 0.856509 | 0.669456 | 0.934253 | 0.987552 | 157 | 23 | 3 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_synthetic_checkpoint_eval\but_clean_fold2_bsold_ep03_on_ptb_metrics.csv`