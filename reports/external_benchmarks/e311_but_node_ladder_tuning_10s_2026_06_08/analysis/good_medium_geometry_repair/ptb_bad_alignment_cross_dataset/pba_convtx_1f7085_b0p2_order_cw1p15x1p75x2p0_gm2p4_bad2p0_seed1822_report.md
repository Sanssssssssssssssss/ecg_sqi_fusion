# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p2_order_cw1p15x1p75x2p0_gm2p4_bad2p0_seed1822 | ptb_val | 0.993019 | 0.991689 | 0.985258 | 1.000000 | 0.976562 | 6 | 0 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p2_order_cw1p15x1p75x2p0_gm2p4_bad2p0_seed1822 | but_clean_margin_ge_5s_drop_outlier_val | 0.867669 | 0.460747 | 0.869565 | 0.860465 | 0.000000 | 81 | 6 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p2_order_cw1p15x1p75x2p0_gm2p4_bad2p0_seed1822 | ptb_synthetic_test | 0.984111 | 0.983079 | 0.949791 | 0.998377 | 0.979253 | 24 | 2 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p2_order_cw1p15x1p75x2p0_gm2p4_bad2p0_seed1822 | but_clean_margin_ge_5s_drop_outlier_test | 0.891841 | 0.919705 | 0.963147 | 0.851228 | 1.000000 | 37 | 309 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank1f7085_b0p2_order_cw1p15x1p75x2p0_gm2p4_bad2p0_seed1822`: best_epoch=4, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=290, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_ff3297e3ede4e5c9\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_1f7085_b0p2_order_cw1p15x1p75x2p0_gm2p4_bad2p0_seed1822_metrics.csv`