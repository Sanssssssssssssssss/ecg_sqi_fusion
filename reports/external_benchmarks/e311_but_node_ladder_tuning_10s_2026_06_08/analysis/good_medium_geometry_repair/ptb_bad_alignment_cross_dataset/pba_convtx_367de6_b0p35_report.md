# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. Original BUT is report-only.

## Why This Exists

Existing bad-gap diagnostics show BUT record111 bad outlier stress differs from PTB synthetic bad in flat/contact/low-amplitude ratios, low detail/zcr, and large baseline span. This run tests a small PTB-only bad-stress augmentation plus non-bad guards.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank367de6_b0p35 | ptb_val | 0.986620 | 0.984064 | 0.972973 | 0.992424 | 0.984375 | 11 | 0 | 4 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank367de6_b0p35 | ptb_synthetic_test | 0.986674 | 0.986562 | 0.951883 | 0.999188 | 0.991701 | 23 | 1 | 2 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank367de6_b0p35 | but_clean_margin_ge_5s_keep_outlier_test | 0.736921 | 0.619993 | 0.655195 | 0.826643 | 0.418006 | 1060 | 384 | 177 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank367de6_b0p35 | but_original_test_all_10s+ | 0.717707 | 0.596314 | 0.634341 | 0.819928 | 0.355231 | 1312 | 439 | 261 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank367de6_b0p35 | but_original_all_10s+ | 0.819365 | 0.835251 | 0.805492 | 0.783779 | 0.935667 | 3269 | 1909 | 336 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank367de6_b0p35 | but_bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank367de6_b0p35 | but_bad_outlier_stress | 0.092466 | 0.056426 | 0.000000 | 0.000000 | 0.092466 | 0 | 0 | 261 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank367de6_b0p35`: best_epoch=3, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1009, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank367de6_b0p35\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Improvement in BUT bad-stress is useful only if good/medium do not collapse.
- Original BUT metrics remain explanatory stress reports, not selection metrics.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_367de6_b0p35_metrics.csv`