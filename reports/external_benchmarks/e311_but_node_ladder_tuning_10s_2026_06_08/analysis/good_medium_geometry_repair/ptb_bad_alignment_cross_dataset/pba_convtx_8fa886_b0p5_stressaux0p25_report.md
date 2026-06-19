# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. Original BUT is report-only.

## Why This Exists

Existing bad-gap diagnostics show BUT record111 bad outlier stress differs from PTB synthetic bad in flat/contact/low-amplitude ratios, low detail/zcr, and large baseline span. This run tests a small PTB-only bad-stress augmentation plus non-bad guards.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p25_distbank8fa886_b0p5 | ptb_val | 0.972659 | 0.966150 | 0.975430 | 0.967803 | 0.988281 | 10 | 3 | 3 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p25_distbank8fa886_b0p5 | ptb_synthetic_test | 0.980523 | 0.977733 | 0.956067 | 0.990260 | 0.979253 | 21 | 6 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p25_distbank8fa886_b0p5 | but_clean_margin_ge_5s_keep_outlier_test | 0.550808 | 0.469160 | 0.669481 | 0.465866 | 0.443730 | 973 | 1515 | 162 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p25_distbank8fa886_b0p5 | but_original_test_all_10s+ | 0.547363 | 0.466254 | 0.651923 | 0.474243 | 0.408759 | 1191 | 1650 | 223 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p25_distbank8fa886_b0p5 | but_original_all_10s+ | 0.745388 | 0.742482 | 0.875433 | 0.439970 | 0.940208 | 1988 | 5220 | 295 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p25_distbank8fa886_b0p5 | but_bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p25_distbank8fa886_b0p5 | but_bad_outlier_stress | 0.167808 | 0.095797 | 0.000000 | 0.000000 | 0.167808 | 0 | 0 | 223 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p25_distbank8fa886_b0p5`: best_epoch=2, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=726, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p25_distbank8fa886_b0p5\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Improvement in BUT bad-stress is useful only if good/medium do not collapse.
- Original BUT metrics remain explanatory stress reports, not selection metrics.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_8fa886_b0p5_stressaux0p25_metrics.csv`