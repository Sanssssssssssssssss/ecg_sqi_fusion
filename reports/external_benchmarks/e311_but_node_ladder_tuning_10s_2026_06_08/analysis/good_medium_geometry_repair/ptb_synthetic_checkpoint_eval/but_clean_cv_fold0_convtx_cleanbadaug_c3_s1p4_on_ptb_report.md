# Checkpoint On PTB Synthetic

Read-only evaluation of an existing waveform-only checkpoint on PTB synthetic val/test splits.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold0\dualview_convtx_hier_cleanbadaug_c3_s1p4\ckpt_best.pt`

| Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_val | 0.746946 | 0.639440 | 0.879607 | 0.821970 | 0.226562 | 49 | 91 | 198 |
| ptb_synthetic_test | 0.801640 | 0.647479 | 0.880753 | 0.896916 | 0.157676 | 57 | 95 | 203 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_synthetic_checkpoint_eval\but_clean_cv_fold0_convtx_cleanbadaug_c3_s1p4_on_ptb_metrics.csv`