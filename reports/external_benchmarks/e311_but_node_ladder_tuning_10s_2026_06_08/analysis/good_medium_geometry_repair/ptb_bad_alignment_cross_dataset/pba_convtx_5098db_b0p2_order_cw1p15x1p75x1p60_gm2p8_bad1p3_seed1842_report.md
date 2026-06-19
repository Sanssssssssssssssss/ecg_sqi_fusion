# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p2_order_cw1p15x1p75x1p60_gm2p8_bad1p3_seed1842 | ptb_val | 0.993601 | 0.992260 | 0.987715 | 1.000000 | 0.976562 | 5 | 0 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p2_order_cw1p15x1p75x1p60_gm2p8_bad1p3_seed1842 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.711994 | 0.498646 | 0.979455 | 0.697115 | 0.000000 | 46 | 378 | 815 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p2_order_cw1p15x1p75x1p60_gm2p8_bad1p3_seed1842 | ptb_synthetic_test | 0.989749 | 0.988527 | 0.972803 | 0.998377 | 0.979253 | 13 | 2 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p2_order_cw1p15x1p75x1p60_gm2p8_bad1p3_seed1842 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.712697 | 0.498721 | 0.983512 | 0.692430 | 0.000000 | 37 | 386 | 817 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p2_order_cw1p15x1p75x1p60_gm2p8_bad1p3_seed1842`: best_epoch=3, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=290, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_315b4951deae11a4\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_5098db_b0p2_order_cw1p15x1p75x1p60_gm2p8_bad1p3_seed1842_metrics.csv`