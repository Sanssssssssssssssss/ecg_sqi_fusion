# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p22_artaux0p14_artnbbad0p08_distbankdefdd6_b1p0_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p85x1p8_gm4p0_bad1p85_featwf_robust_no_physical | ptb_val | 0.949389 | 0.948408 | 1.000000 | 0.923295 | 0.976562 | 0 | 70 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p22_artaux0p14_artnbbad0p08_distbankdefdd6_b1p0_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p85x1p8_gm4p0_bad1p85_featwf_robust_no_physical | ptb_combo_stress_val | 0.970414 | 0.971246 | 1.000000 | 0.988000 | 0.928144 | 0 | 0 | 9 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p22_artaux0p14_artnbbad0p08_distbankdefdd6_b1p0_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p85x1p8_gm4p0_bad1p85_featwf_robust_no_physical | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_val | 0.765097 | 0.792016 | 0.653551 | 0.836512 | 0.963035 | 1020 | 187 | 33 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p22_artaux0p14_artnbbad0p08_distbankdefdd6_b1p0_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p85x1p8_gm4p0_bad1p85_featwf_robust_no_physical | ptb_synthetic_test | 0.948744 | 0.940726 | 0.983264 | 0.927760 | 0.987552 | 1 | 65 | 3 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p22_artaux0p14_artnbbad0p08_distbankdefdd6_b1p0_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p85x1p8_gm4p0_bad1p85_featwf_robust_no_physical | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_test | 0.763216 | 0.789882 | 0.657019 | 0.825380 | 0.961278 | 1009 | 209 | 32 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p22_artaux0p14_artnbbad0p08_distbankdefdd6_b1p0_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p85x1p8_gm4p0_bad1p85_featwf_robust_no_physical`: best_epoch=4, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=1287, matched_bank_added=4563, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_44172f29891f48bb\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_44172f29891f48bb\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_defdd6_b1p0_stratified_order_hold0p1_stre_1b5076b3af57_metrics.csv`