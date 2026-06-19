# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. Original BUT is report-only.

## Why This Exists

Existing bad-gap diagnostics show BUT record111 bad outlier stress differs from PTB synthetic bad in flat/contact/low-amplitude ratios, low detail/zcr, and large baseline span. This run tests a small PTB-only bad-stress augmentation plus non-bad guards.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p0 | ptb_val | 0.906923 | 0.893117 | 0.992629 | 0.852273 | 0.996094 | 3 | 26 | 1 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p0 | ptb_synthetic_test | 0.966171 | 0.965719 | 0.983264 | 0.952922 | 1.000000 | 8 | 49 | 0 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p0 | but_clean_margin_ge_5s_keep_outlier_test | 0.535470 | 0.462620 | 0.637338 | 0.445155 | 0.662379 | 858 | 1046 | 100 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p0 | but_original_test_all_10s+ | 0.523416 | 0.455179 | 0.617308 | 0.437415 | 0.618005 | 1013 | 1131 | 148 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p0 | but_original_all_10s+ | 0.740624 | 0.722793 | 0.859238 | 0.444204 | 0.954210 | 1844 | 4425 | 230 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p0 | but_bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p0 | but_bad_outlier_stress | 0.462329 | 0.210773 | 0.000000 | 0.000000 | 0.462329 | 0 | 0 | 148 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p25 | ptb_val | 0.961024 | 0.951590 | 0.968059 | 0.952652 | 0.984375 | 13 | 0 | 4 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p25 | ptb_synthetic_test | 0.978985 | 0.978078 | 0.930962 | 0.997565 | 0.979253 | 33 | 3 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p25 | but_clean_margin_ge_5s_keep_outlier_test | 0.682827 | 0.497582 | 0.597403 | 0.797750 | 0.083601 | 1233 | 385 | 281 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p25 | but_original_test_all_10s+ | 0.664976 | 0.497125 | 0.575549 | 0.789200 | 0.119221 | 1520 | 441 | 358 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p25 | but_original_all_10s+ | 0.758284 | 0.785626 | 0.657044 | 0.842774 | 0.914853 | 5751 | 1139 | 446 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p25 | but_bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p25 | but_bad_outlier_stress | 0.167808 | 0.095797 | 0.000000 | 0.000000 | 0.167808 | 0 | 0 | 239 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p0`: best_epoch=2, bad_aug_added=390, nonbad_guard_added=1201, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p0\ckpt_best.pt`
- `ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p25`: best_epoch=2, bad_aug_added=398, nonbad_guard_added=1234, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_s1p25\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Improvement in BUT bad-stress is useful only if good/medium do not collapse.
- Original BUT metrics remain explanatory stress reports, not selection metrics.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\ptb_bad_alignment_margin_ge_5s_keep_outlier_dualview_convtx_hier_p30_metrics.csv`