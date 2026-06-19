# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. Original BUT is report-only.

## Why This Exists

Existing bad-gap diagnostics show BUT record111 bad outlier stress differs from PTB synthetic bad in flat/contact/low-amplitude ratios, low detail/zcr, and large baseline span. This run tests a small PTB-only bad-stress augmentation plus non-bad guards.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b0p35 | ptb_val | 0.990111 | 0.988357 | 0.977887 | 0.998106 | 0.976562 | 9 | 0 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b0p35 | ptb_synthetic_test | 0.982060 | 0.979492 | 0.953975 | 0.993506 | 0.979253 | 22 | 3 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b0p35 | but_clean_margin_ge_5s_keep_outlier_test | 0.631334 | 0.539943 | 0.659416 | 0.625671 | 0.424437 | 1038 | 1090 | 172 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b0p35 | but_original_test_all_10s+ | 0.619559 | 0.525486 | 0.641484 | 0.624492 | 0.372263 | 1276 | 1211 | 248 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b0p35 | but_original_all_10s+ | 0.777036 | 0.784414 | 0.868098 | 0.551091 | 0.937748 | 2174 | 4275 | 318 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b0p35 | but_bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b0p35 | but_bad_outlier_stress | 0.116438 | 0.069530 | 0.000000 | 0.000000 | 0.116438 | 0 | 0 | 248 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b0p35`: best_epoch=2, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=448, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_distbank_b0p35\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Improvement in BUT bad-stress is useful only if good/medium do not collapse.
- Original BUT metrics remain explanatory stress reports, not selection metrics.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\ptb_bad_alignment_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_distbank_b0p35_metrics.csv`