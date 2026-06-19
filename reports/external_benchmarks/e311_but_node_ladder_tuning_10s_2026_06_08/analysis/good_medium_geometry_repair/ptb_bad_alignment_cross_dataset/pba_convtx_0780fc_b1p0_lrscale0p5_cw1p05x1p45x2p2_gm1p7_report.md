# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank0780fc_b1p0_lrscale0p5_cw1p05x1p45x2p2_gm1p7 | ptb_val | 0.993601 | 0.992260 | 0.987715 | 1.000000 | 0.976562 | 5 | 0 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank0780fc_b1p0_lrscale0p5_cw1p05x1p45x2p2_gm1p7 | but_clean_margin_ge_5s_drop_outlier_val | 0.951880 | 0.555667 | 0.961353 | 0.837209 | 0.000000 | 24 | 7 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank0780fc_b1p0_lrscale0p5_cw1p05x1p45x2p2_gm1p7 | ptb_synthetic_test | 0.989749 | 0.988527 | 0.972803 | 0.998377 | 0.979253 | 13 | 2 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank0780fc_b1p0_lrscale0p5_cw1p05x1p45x2p2_gm1p7 | but_clean_margin_ge_5s_drop_outlier_test | 0.796499 | 0.537824 | 0.966135 | 0.759750 | 0.000000 | 34 | 499 | 118 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank0780fc_b1p0_lrscale0p5_cw1p05x1p45x2p2_gm1p7`: best_epoch=4, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=972, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_8035e9e1e730fb3a\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_0780fc_b1p0_lrscale0p5_cw1p05x1p45x2p2_gm1p7_metrics.csv`