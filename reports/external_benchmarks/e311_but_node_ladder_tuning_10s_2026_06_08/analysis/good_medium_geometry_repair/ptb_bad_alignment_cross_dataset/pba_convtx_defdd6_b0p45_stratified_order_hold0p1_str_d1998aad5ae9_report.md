# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p06_distbankdefdd6_b0p45_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p75x1p8_gm4p2_bad1p85_featwf | ptb_val | 0.962769 | 0.962820 | 0.992629 | 0.947917 | 0.976562 | 3 | 53 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p06_distbankdefdd6_b0p45_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p75x1p8_gm4p2_bad1p85_featwf | ptb_combo_stress_val | 0.970414 | 0.969904 | 1.000000 | 1.000000 | 0.910180 | 0 | 0 | 11 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p06_distbankdefdd6_b0p45_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p75x1p8_gm4p2_bad1p85_featwf | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_val | 0.806209 | 0.811985 | 0.911637 | 0.550409 | 0.955253 | 265 | 781 | 42 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p06_distbankdefdd6_b0p45_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p75x1p8_gm4p2_bad1p85_featwf | ptb_synthetic_test | 0.961558 | 0.960602 | 0.970711 | 0.954545 | 0.979253 | 14 | 50 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p06_distbankdefdd6_b0p45_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p75x1p8_gm4p2_bad1p85_featwf | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_test | 0.801462 | 0.806349 | 0.914172 | 0.531453 | 0.955470 | 257 | 820 | 42 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p06_distbankdefdd6_b0p45_stratified_order_hold0p1_selclean_lrscale0p65_cw1x1p75x1p8_gm4p2_bad1p85_featwf`: best_epoch=4, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=956, matched_bank_added=2053, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_9803414d0b800389\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_9803414d0b800389\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_defdd6_b0p45_stratified_order_hold0p1_str_d1998aad5ae9_metrics.csv`