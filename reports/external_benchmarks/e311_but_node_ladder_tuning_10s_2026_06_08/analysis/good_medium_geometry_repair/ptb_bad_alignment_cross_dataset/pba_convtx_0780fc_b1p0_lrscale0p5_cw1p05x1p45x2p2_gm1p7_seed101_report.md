# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank0780fc_b1p0_lrscale0p5_cw1p05x1p45x2p2_gm1p7_seed101 | ptb_val | 0.978476 | 0.977839 | 0.992629 | 0.973485 | 0.976562 | 3 | 28 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank0780fc_b1p0_lrscale0p5_cw1p05x1p45x2p2_gm1p7_seed101 | but_clean_margin_ge_5s_drop_outlier_val | 0.966917 | 0.567054 | 0.990338 | 0.651163 | 0.000000 | 6 | 15 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank0780fc_b1p0_lrscale0p5_cw1p05x1p45x2p2_gm1p7_seed101 | ptb_synthetic_test | 0.954382 | 0.957103 | 0.997908 | 0.932630 | 0.979253 | 1 | 83 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank0780fc_b1p0_lrscale0p5_cw1p05x1p45x2p2_gm1p7_seed101 | but_clean_margin_ge_5s_drop_outlier_test | 0.778993 | 0.801875 | 0.998008 | 0.674531 | 0.754237 | 2 | 676 | 29 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank0780fc_b1p0_lrscale0p5_cw1p05x1p45x2p2_gm1p7_seed101`: best_epoch=6, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=972, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_c7d09dd5ff1fb637\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_0780fc_b1p0_lrscale0p5_cw1p05x1p45x2p2_gm1p7_seed101_metrics.csv`