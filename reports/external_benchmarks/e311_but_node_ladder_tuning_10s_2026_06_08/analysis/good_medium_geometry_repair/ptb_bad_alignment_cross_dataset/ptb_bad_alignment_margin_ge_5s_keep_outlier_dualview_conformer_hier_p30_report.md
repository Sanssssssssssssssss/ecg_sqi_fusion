# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. Original BUT is report-only.

## Why This Exists

Existing bad-gap diagnostics show BUT record111 bad outlier stress differs from PTB synthetic bad in flat/contact/low-amplitude ratios, low detail/zcr, and large baseline span. This run tests a small PTB-only bad-stress augmentation plus non-bad guards.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p15 | ptb_val | 0.991856 | 0.990325 | 0.982801 | 0.998106 | 0.980469 | 7 | 0 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p15 | ptb_synthetic_test | 0.985136 | 0.984470 | 0.958159 | 0.995130 | 0.987552 | 20 | 5 | 3 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p15 | but_clean_margin_ge_5s_keep_outlier_test | 0.589838 | 0.493362 | 0.595455 | 0.594988 | 0.469453 | 1160 | 851 | 152 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p15 | but_original_test_all_10s+ | 0.578153 | 0.485362 | 0.577747 | 0.591957 | 0.433090 | 1402 | 934 | 212 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p15 | but_original_all_10s+ | 0.761015 | 0.758844 | 0.848853 | 0.529262 | 0.943803 | 2301 | 4021 | 275 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p15 | but_bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p15 | but_bad_outlier_stress | 0.202055 | 0.112061 | 0.000000 | 0.000000 | 0.202055 | 0 | 0 | 212 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p15`: best_epoch=2, bad_aug_added=412, nonbad_guard_added=1239, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_s1p15\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Improvement in BUT bad-stress is useful only if good/medium do not collapse.
- Original BUT metrics remain explanatory stress reports, not selection metrics.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\ptb_bad_alignment_margin_ge_5s_keep_outlier_dualview_conformer_hier_p30_metrics.csv`