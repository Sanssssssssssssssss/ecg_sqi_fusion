# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p15x1p45x2_gm1p8_bad2p0_seed808 | ptb_val | 0.988947 | 0.987759 | 0.990172 | 0.991477 | 0.976562 | 4 | 9 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p15x1p45x2_gm1p8_bad2p0_seed808 | but_clean_margin_ge_5s_drop_outlier_val | 0.933835 | 0.532589 | 0.938808 | 0.883721 | 0.000000 | 38 | 5 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p15x1p45x2_gm1p8_bad2p0_seed808 | ptb_synthetic_test | 0.988211 | 0.987125 | 0.981172 | 0.992695 | 0.979253 | 9 | 9 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p15x1p45x2_gm1p8_bad2p0_seed808 | but_clean_margin_ge_5s_drop_outlier_test | 0.908409 | 0.929125 | 0.977092 | 0.870968 | 0.983051 | 23 | 268 | 2 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p15x1p45x2_gm1p8_bad2p0_seed808`: best_epoch=4, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=640, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_ac2388cf234c8ec7\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_39ff5d_b0p5_cw1p15x1p45x2_gm1p8_bad2p0_seed808_metrics.csv`