# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p2_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1821 | ptb_val | 0.994183 | 0.992847 | 0.992629 | 0.998106 | 0.980469 | 3 | 1 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p2_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1821 | but_clean_margin_ge_5s_drop_outlier_val | 0.939850 | 0.542937 | 0.943639 | 0.906977 | 0.000000 | 35 | 4 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p2_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1821 | ptb_synthetic_test | 0.990774 | 0.989881 | 0.981172 | 0.995942 | 0.983402 | 9 | 5 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p2_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1821 | but_clean_margin_ge_5s_drop_outlier_test | 0.919037 | 0.939150 | 0.974104 | 0.887819 | 1.000000 | 26 | 233 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p2_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1821`: best_epoch=5, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=290, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_cc816eafaaf54e36\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_1f7085_b0p2_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1821_metrics.csv`