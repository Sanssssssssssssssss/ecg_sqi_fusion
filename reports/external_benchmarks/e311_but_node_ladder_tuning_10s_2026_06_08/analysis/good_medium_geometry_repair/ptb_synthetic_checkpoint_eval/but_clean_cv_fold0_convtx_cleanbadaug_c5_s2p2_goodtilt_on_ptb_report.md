# Checkpoint On PTB Synthetic

Read-only evaluation of an existing waveform-only checkpoint on PTB synthetic val/test splits.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold0\dualview_convtx_hier_cleanbadaug_c5_s2p2_goodtilt\ckpt_best.pt`

| Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_val | 0.853985 | 0.842680 | 0.869779 | 0.816288 | 0.984375 | 53 | 89 | 4 |
| ptb_synthetic_test | 0.880574 | 0.877032 | 0.849372 | 0.872565 | 0.983402 | 69 | 124 | 4 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_synthetic_checkpoint_eval\but_clean_cv_fold0_convtx_cleanbadaug_c5_s2p2_goodtilt_on_ptb_metrics.csv`