# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank04f776_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_avgE2x3x4x6_featwf | ptb_val | 0.986620 | 0.984196 | 0.968059 | 0.996212 | 0.976562 | 12 | 1 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank04f776_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_avgE2x3x4x6_featwf | ptb_combo_stress_val | 0.872109 | 0.862065 | 1.000000 | 0.991968 | 0.610169 | 0 | 0 | 75 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank04f776_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_avgE2x3x4x6_featwf | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_val | 0.827874 | 0.840038 | 0.841947 | 0.729700 | 0.962062 | 451 | 436 | 39 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank04f776_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_avgE2x3x4x6_featwf | ptb_synthetic_test | 0.976935 | 0.966965 | 0.966527 | 0.979708 | 0.983402 | 9 | 5 | 4 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank04f776_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_avgE2x3x4x6_featwf | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_test | 0.828999 | 0.841095 | 0.850965 | 0.718547 | 0.962246 | 424 | 467 | 38 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank04f776_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_avgE2x3x4x6_featwf`: best_epoch=-1, averaged_epochs=[2, 3, 4, 6], bad_aug_added=0, nonbad_guard_added=1291, matched_bank_added=6621, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_aaf8d436a1958fec\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_aaf8d436a1958fec\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_04f776_b1p0_stratified_order_hold0p1_stre_2d93f54e1bd4_metrics.csv`