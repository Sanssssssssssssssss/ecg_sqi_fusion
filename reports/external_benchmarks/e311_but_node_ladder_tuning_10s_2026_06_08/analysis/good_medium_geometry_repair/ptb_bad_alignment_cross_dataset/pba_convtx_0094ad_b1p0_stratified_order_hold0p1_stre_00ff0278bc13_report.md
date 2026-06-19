# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank0094ad_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_avgE2x3x4x6_featwf | ptb_val | 0.969750 | 0.960673 | 0.980344 | 0.964015 | 0.976562 | 5 | 3 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank0094ad_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_avgE2x3x4x6_featwf | ptb_combo_stress_val | 0.882674 | 0.872899 | 1.000000 | 1.000000 | 0.634043 | 0 | 0 | 60 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank0094ad_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_avgE2x3x4x6_featwf | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_val | 0.820710 | 0.829173 | 0.855952 | 0.686104 | 0.958171 | 388 | 511 | 43 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank0094ad_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_avgE2x3x4x6_featwf | ptb_synthetic_test | 0.961046 | 0.942390 | 0.953975 | 0.959416 | 0.983402 | 8 | 3 | 4 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank0094ad_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_avgE2x3x4x6_featwf | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_test | 0.822540 | 0.830545 | 0.862608 | 0.681128 | 0.958374 | 366 | 527 | 42 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank0094ad_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_avgE2x3x4x6_featwf`: best_epoch=-1, averaged_epochs=[2, 3, 4, 6], bad_aug_added=0, nonbad_guard_added=1270, matched_bank_added=6615, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_aa920ffcf5d98737\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_aa920ffcf5d98737\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_0094ad_b1p0_stratified_order_hold0p1_stre_00ff0278bc13_metrics.csv`