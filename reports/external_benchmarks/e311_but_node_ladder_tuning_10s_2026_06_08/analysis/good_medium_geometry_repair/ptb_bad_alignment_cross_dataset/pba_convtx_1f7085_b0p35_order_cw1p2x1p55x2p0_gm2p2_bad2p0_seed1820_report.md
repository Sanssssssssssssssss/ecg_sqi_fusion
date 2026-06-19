# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p35_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1820 | ptb_val | 0.987784 | 0.986508 | 0.963145 | 1.000000 | 0.976562 | 15 | 0 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p35_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1820 | but_clean_margin_ge_5s_drop_outlier_val | 0.926316 | 0.526718 | 0.927536 | 0.930233 | 0.000000 | 45 | 3 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p35_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1820 | ptb_synthetic_test | 0.960533 | 0.959362 | 0.853556 | 0.998377 | 0.979253 | 70 | 2 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p35_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1820 | but_clean_margin_ge_5s_drop_outlier_test | 0.919975 | 0.938609 | 0.927291 | 0.911892 | 1.000000 | 73 | 183 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p35_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1820`: best_epoch=3, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=507, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_395b4a33300ff70f\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_1f7085_b0p35_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1820_metrics.csv`