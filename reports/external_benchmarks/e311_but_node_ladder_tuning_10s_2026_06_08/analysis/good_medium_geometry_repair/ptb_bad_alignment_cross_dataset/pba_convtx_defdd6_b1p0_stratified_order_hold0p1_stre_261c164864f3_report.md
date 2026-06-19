# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p22_artaux0p14_artnbbad0p08_distbankdefdd6_b1p0_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p85x1p8_gm4p0_bad1p85_featwf | ptb_val | 0.986620 | 0.984705 | 0.972973 | 0.994318 | 0.976562 | 11 | 3 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p22_artaux0p14_artnbbad0p08_distbankdefdd6_b1p0_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p85x1p8_gm4p0_bad1p85_featwf | ptb_combo_stress_val | 0.974359 | 0.975662 | 1.000000 | 1.000000 | 0.922156 | 0 | 0 | 11 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p22_artaux0p14_artnbbad0p08_distbankdefdd6_b1p0_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p85x1p8_gm4p0_bad1p85_featwf | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_val | 0.838451 | 0.851822 | 0.837613 | 0.774387 | 0.955253 | 484 | 356 | 43 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p22_artaux0p14_artnbbad0p08_distbankdefdd6_b1p0_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p85x1p8_gm4p0_bad1p85_featwf | ptb_synthetic_test | 0.992312 | 0.990634 | 0.983264 | 0.998377 | 0.979253 | 8 | 1 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p22_artaux0p14_artnbbad0p08_distbankdefdd6_b1p0_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p85x1p8_gm4p0_bad1p85_featwf | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_test | 0.836138 | 0.849260 | 0.845309 | 0.754881 | 0.954501 | 461 | 397 | 43 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p22_artaux0p14_artnbbad0p08_distbankdefdd6_b1p0_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p85x1p8_gm4p0_bad1p85_featwf`: best_epoch=5, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=1287, matched_bank_added=4563, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_a340eda2044ab0ea\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_a340eda2044ab0ea\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_defdd6_b1p0_stratified_order_hold0p1_stre_261c164864f3_metrics.csv`