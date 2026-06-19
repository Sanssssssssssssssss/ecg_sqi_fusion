# Checkpoint On PTB Synthetic

Read-only evaluation of an existing waveform-only checkpoint on PTB synthetic val/test splits.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold3\dualview_convtx_hier_ptbinit_b0p5_replay025\ckpt_best.pt`

| Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_val | 0.977894 | 0.971470 | 0.990172 | 0.970644 | 0.988281 | 4 | 1 | 3 |
| ptb_synthetic_test | 0.990261 | 0.990047 | 0.968619 | 0.998377 | 0.991701 | 15 | 2 | 2 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_synthetic_checkpoint_eval\but_clean_cv_fold3_convtx_ptbinit_replay025_on_ptb_metrics.csv`