# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. Original BUT is report-only.

## Why This Exists

Existing bad-gap diagnostics show BUT record111 bad outlier stress differs from PTB synthetic bad in flat/contact/low-amplitude ratios, low detail/zcr, and large baseline span. This run tests a small PTB-only bad-stress augmentation plus non-bad guards.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s0p75 | ptb_val | 0.978476 | 0.977111 | 0.923833 | 1.000000 | 0.976562 | 31 | 0 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s0p75 | ptb_synthetic_test | 0.981548 | 0.980555 | 0.937238 | 0.999188 | 0.979253 | 30 | 1 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s0p75 | but_clean_margin_ge_5s_keep_outlier_test | 0.714599 | 0.486172 | 0.573377 | 0.882383 | 0.003215 | 1313 | 300 | 305 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s0p75 | but_original_test_all_10s+ | 0.698124 | 0.485999 | 0.554945 | 0.878446 | 0.024331 | 1610 | 341 | 396 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s0p75 | but_original_all_10s+ | 0.661124 | 0.478398 | 0.767588 | 0.818216 | 0.001892 | 3924 | 1710 | 5270 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s0p75 | but_bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s0p75 | but_bad_outlier_stress | 0.034247 | 0.022075 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 277 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s1p15 | ptb_val | 0.908668 | 0.897437 | 0.990172 | 0.856061 | 0.996094 | 4 | 60 | 1 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s1p15 | ptb_synthetic_test | 0.944131 | 0.938680 | 0.976987 | 0.922890 | 0.987552 | 11 | 67 | 3 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s1p15 | but_clean_margin_ge_5s_keep_outlier_test | 0.557929 | 0.480697 | 0.770455 | 0.377653 | 0.720257 | 510 | 1453 | 73 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s1p15 | but_original_test_all_10s+ | 0.550077 | 0.477733 | 0.750000 | 0.372119 | 0.695864 | 641 | 1603 | 105 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s1p15 | but_original_all_10s+ | 0.752701 | 0.730967 | 0.901719 | 0.407885 | 0.965563 | 1292 | 4951 | 161 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s1p15 | but_bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s1p15 | but_bad_outlier_stress | 0.571918 | 0.242556 | 0.000000 | 0.000000 | 0.571918 | 0 | 0 | 105 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s0p75`: best_epoch=2, bad_aug_added=789, nonbad_guard_added=749, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s0p75\ckpt_best.pt`
- `ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s1p15`: best_epoch=2, bad_aug_added=788, nonbad_guard_added=811, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_s1p15\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Improvement in BUT bad-stress is useful only if good/medium do not collapse.
- Original BUT metrics remain explanatory stress reports, not selection metrics.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\ptb_bad_alignment_margin_ge_5s_keep_outlier_dualview_convtx_hier_p60_metrics.csv`