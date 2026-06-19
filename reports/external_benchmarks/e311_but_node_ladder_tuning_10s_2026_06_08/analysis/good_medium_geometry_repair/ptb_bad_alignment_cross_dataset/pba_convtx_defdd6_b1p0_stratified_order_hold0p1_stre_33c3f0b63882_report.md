# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p15_artaux0p25_artnbbad0p35_distbankdefdd6_b1p0_stratified_order_hold0p1_selstressbal_lrscale0p75_cw1p1x1p55x1p55_gm3p0_bad1p55_featwf | ptb_val | 0.985457 | 0.984175 | 0.992629 | 0.984848 | 0.976562 | 3 | 15 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p15_artaux0p25_artnbbad0p35_distbankdefdd6_b1p0_stratified_order_hold0p1_selstressbal_lrscale0p75_cw1p1x1p55x1p55_gm3p0_bad1p55_featwf | ptb_combo_stress_val | 0.970414 | 0.971049 | 1.000000 | 1.000000 | 0.910180 | 0 | 0 | 12 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p15_artaux0p25_artnbbad0p35_distbankdefdd6_b1p0_stratified_order_hold0p1_selstressbal_lrscale0p75_cw1p1x1p55x1p55_gm3p0_bad1p55_featwf | but_clean_margin_ge_5s_keep_outlier_val | 0.853683 | 0.733100 | 0.858989 | 0.883333 | 0.775000 | 120 | 4 | 18 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p15_artaux0p25_artnbbad0p35_distbankdefdd6_b1p0_stratified_order_hold0p1_selstressbal_lrscale0p75_cw1p1x1p55x1p55_gm3p0_bad1p55_featwf | ptb_synthetic_test | 0.975397 | 0.975031 | 0.985356 | 0.970779 | 0.979253 | 7 | 35 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p15_artaux0p25_artnbbad0p35_distbankdefdd6_b1p0_stratified_order_hold0p1_selstressbal_lrscale0p75_cw1p1x1p55x1p55_gm3p0_bad1p55_featwf | but_clean_margin_ge_5s_keep_outlier_test | 0.698850 | 0.602642 | 0.689286 | 0.730248 | 0.398714 | 955 | 846 | 182 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_dualview_convtx_hier_p00_s0p0_stressaux0p15_artaux0p25_artnbbad0p35_distbankdefdd6_b1p0_stratified_order_hold0p1_selstressbal_lrscale0p75_cw1p1x1p55x1p55_gm3p0_bad1p55_featwf`: best_epoch=3, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=2731, matched_bank_added=4563, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_ec612c1ff79a17c7\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_ec612c1ff79a17c7\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_defdd6_b1p0_stratified_order_hold0p1_stre_33c3f0b63882_metrics.csv`