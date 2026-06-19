# Checkpoint On PTB Synthetic

Read-only evaluation of an existing waveform-only checkpoint on PTB synthetic val/test splits.

Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\clean_but_dualview_hier_transformer\margin_ge_5s_drop_outlier_cv_seed20260619_fold0\dualview_convtx_hier_clean_gmptbaug_g1m1_b5s2p2_badkeep\ckpt_best.pt`

| Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_synthetic_val | 0.791158 | 0.748085 | 0.891892 | 0.804924 | 0.574219 | 44 | 125 | 109 |
| ptb_synthetic_test | 0.803178 | 0.765884 | 0.859833 | 0.825487 | 0.576763 | 67 | 198 | 102 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_synthetic_checkpoint_eval\but_clean_cv_fold0_convtx_clean_gmptbaug_g1m1_b5s2p2_badkeep_on_ptb_metrics.csv`