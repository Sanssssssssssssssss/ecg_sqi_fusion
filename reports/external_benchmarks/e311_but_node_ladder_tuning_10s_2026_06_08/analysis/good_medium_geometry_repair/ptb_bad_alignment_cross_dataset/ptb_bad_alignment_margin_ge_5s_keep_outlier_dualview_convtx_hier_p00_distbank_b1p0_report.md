# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. Original BUT is report-only.

## Why This Exists

Existing bad-gap diagnostics show BUT record111 bad outlier stress differs from PTB synthetic bad in flat/contact/low-amplitude ratios, low detail/zcr, and large baseline span. This run tests a small PTB-only bad-stress augmentation plus non-bad guards.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b1p0 | ptb_val | 0.987202 | 0.983171 | 1.000000 | 0.982955 | 0.984375 | 0 | 3 | 4 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b1p0 | ptb_synthetic_test | 0.988211 | 0.987574 | 0.997908 | 0.985390 | 0.983402 | 1 | 18 | 4 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b1p0 | but_clean_margin_ge_5s_keep_outlier_test | 0.675842 | 0.589918 | 0.700649 | 0.677065 | 0.414791 | 916 | 1051 | 167 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b1p0 | but_original_test_all_10s+ | 0.665094 | 0.568103 | 0.682692 | 0.681428 | 0.333333 | 1144 | 1169 | 250 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b1p0 | but_original_all_10s+ | 0.802616 | 0.814612 | 0.883119 | 0.608487 | 0.933396 | 1981 | 3915 | 327 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b1p0 | but_bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b1p0 | but_bad_outlier_stress | 0.061644 | 0.038710 | 0.000000 | 0.000000 | 0.061644 | 0 | 0 | 250 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b1p0`: best_epoch=3, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1451, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b1p0\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Improvement in BUT bad-stress is useful only if good/medium do not collapse.
- Original BUT metrics remain explanatory stress reports, not selection metrics.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\ptb_bad_alignment_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_distbank_b1p0_metrics.csv`