# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p25_order_cw1p18x1p70x1p70_gm2p6_bad1p5_seed1843 | ptb_val | 0.993601 | 0.991368 | 0.997543 | 0.994318 | 0.984375 | 1 | 0 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p25_order_cw1p18x1p70x1p70_gm2p6_bad1p5_seed1843 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.841934 | 0.804793 | 0.974542 | 0.758013 | 0.606135 | 57 | 302 | 321 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p25_order_cw1p18x1p70x1p70_gm2p6_bad1p5_seed1843 | ptb_synthetic_test | 0.990774 | 0.989856 | 0.974895 | 0.998377 | 0.983402 | 12 | 2 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p25_order_cw1p18x1p70x1p70_gm2p6_bad1p5_seed1843 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.844300 | 0.804866 | 0.976827 | 0.771315 | 0.592411 | 52 | 287 | 333 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p25_order_cw1p18x1p70x1p70_gm2p6_bad1p5_seed1843`: best_epoch=5, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=362, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_8d6f273f59bf55f8\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_5098db_b0p25_order_cw1p18x1p70x1p70_gm2p6_bad1p5_seed1843_metrics.csv`