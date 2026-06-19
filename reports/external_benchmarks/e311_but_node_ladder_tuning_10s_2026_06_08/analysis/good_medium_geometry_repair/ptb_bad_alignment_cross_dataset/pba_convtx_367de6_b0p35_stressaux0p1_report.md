# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. Original BUT is report-only.

## Why This Exists

Existing bad-gap diagnostics show BUT record111 bad outlier stress differs from PTB synthetic bad in flat/contact/low-amplitude ratios, low detail/zcr, and large baseline span. This run tests a small PTB-only bad-stress augmentation plus non-bad guards.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p1_distbank367de6_b0p35 | ptb_val | 0.989529 | 0.986690 | 0.990172 | 0.991477 | 0.980469 | 4 | 1 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p1_distbank367de6_b0p35 | ptb_synthetic_test | 0.988211 | 0.987168 | 0.987448 | 0.989448 | 0.983402 | 6 | 12 | 4 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p1_distbank367de6_b0p35 | but_clean_margin_ge_5s_keep_outlier_test | 0.676390 | 0.565832 | 0.776948 | 0.614932 | 0.453376 | 669 | 984 | 158 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p1_distbank367de6_b0p35 | but_original_test_all_10s+ | 0.660611 | 0.551202 | 0.756593 | 0.605287 | 0.406326 | 842 | 1114 | 228 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p1_distbank367de6_b0p35 | but_original_all_10s+ | 0.800249 | 0.798897 | 0.898785 | 0.573203 | 0.939073 | 1588 | 3848 | 306 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p1_distbank367de6_b0p35 | but_bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p1_distbank367de6_b0p35 | but_bad_outlier_stress | 0.164384 | 0.094118 | 0.000000 | 0.000000 | 0.164384 | 0 | 0 | 228 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p1_distbank367de6_b0p35`: best_epoch=2, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1009, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p1_distbank367de6_b0p35\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Improvement in BUT bad-stress is useful only if good/medium do not collapse.
- Original BUT metrics remain explanatory stress reports, not selection metrics.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_367de6_b0p35_stressaux0p1_metrics.csv`