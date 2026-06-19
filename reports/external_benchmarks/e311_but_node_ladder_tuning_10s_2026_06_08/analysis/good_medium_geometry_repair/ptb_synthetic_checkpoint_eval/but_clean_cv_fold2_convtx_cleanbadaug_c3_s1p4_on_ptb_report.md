# Checkpoint On PTB Synthetic

Read-only evaluation of an existing waveform-only checkpoint on PTB synthetic val/test splits.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold2\dualview_convtx_hier_cleanbadaug_c3_s1p4\ckpt_best.pt`

| Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_val | 0.744619 | 0.556308 | 0.889435 | 0.868371 | 0.003906 | 45 | 77 | 255 |
| ptb_synthetic_test | 0.791389 | 0.560410 | 0.866109 | 0.917208 | 0.000000 | 64 | 101 | 241 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_synthetic_checkpoint_eval\but_clean_cv_fold2_convtx_cleanbadaug_c3_s1p4_on_ptb_metrics.csv`