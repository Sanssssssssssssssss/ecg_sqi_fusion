# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbankfb3b3b_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_featwf | ptb_val | 0.987784 | 0.986380 | 0.987715 | 0.990530 | 0.976562 | 5 | 9 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbankfb3b3b_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_featwf | ptb_combo_stress_val | 0.825613 | 0.801566 | 1.000000 | 1.000000 | 0.455319 | 0 | 0 | 98 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbankfb3b3b_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_featwf | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_val | 0.807233 | 0.814108 | 0.907302 | 0.560763 | 0.955253 | 265 | 775 | 45 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbankfb3b3b_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_featwf | ptb_synthetic_test | 0.980010 | 0.979324 | 0.958159 | 0.988636 | 0.979253 | 20 | 14 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbankfb3b3b_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_featwf | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_test | 0.802992 | 0.807368 | 0.917498 | 0.531453 | 0.954501 | 231 | 833 | 46 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbankfb3b3b_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_featwf`: best_epoch=2, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=1245, matched_bank_added=6618, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_9faa0b751b6517f8\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_9faa0b751b6517f8\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_fb3b3b_b1p0_stratified_order_hold0p1_stre_f83593c51cc4_metrics.csv`