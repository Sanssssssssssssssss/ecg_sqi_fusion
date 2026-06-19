# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. Original BUT is report-only.

## Why This Exists

Existing bad-gap diagnostics show BUT record111 bad outlier stress differs from PTB synthetic bad in flat/contact/low-amplitude ratios, low detail/zcr, and large baseline span. This run tests a small PTB-only bad-stress augmentation plus non-bad guards.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p0_stressaux0p25 | ptb_val | 0.979639 | 0.978907 | 0.990172 | 0.976326 | 0.976562 | 4 | 25 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p0_stressaux0p25 | ptb_synthetic_test | 0.975910 | 0.975813 | 0.981172 | 0.973214 | 0.979253 | 9 | 33 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p0_stressaux0p25 | but_clean_margin_ge_5s_keep_outlier_test | 0.688989 | 0.485187 | 0.721104 | 0.716696 | 0.022508 | 858 | 877 | 277 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p0_stressaux0p25 | but_original_test_all_10s+ | 0.673705 | 0.482878 | 0.700000 | 0.711026 | 0.038929 | 1080 | 994 | 360 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p0_stressaux0p25 | but_original_all_10s+ | 0.791115 | 0.800258 | 0.874905 | 0.602371 | 0.900473 | 2106 | 3929 | 489 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p0_stressaux0p25 | but_bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p0_stressaux0p25 | but_bad_outlier_stress | 0.054795 | 0.034632 | 0.000000 | 0.000000 | 0.054795 | 0 | 0 | 241 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p0_stressaux0p25`: best_epoch=2, bad_aug_added=383, nonbad_guard_added=1241, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p0_stressaux0p25\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Improvement in BUT bad-stress is useful only if good/medium do not collapse.
- Original BUT metrics remain explanatory stress reports, not selection metrics.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\ptb_bad_alignment_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_stressaux0p25_metrics.csv`