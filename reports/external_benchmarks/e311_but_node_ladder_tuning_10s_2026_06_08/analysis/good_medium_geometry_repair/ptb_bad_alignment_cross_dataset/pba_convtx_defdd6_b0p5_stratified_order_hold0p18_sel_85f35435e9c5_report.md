# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbankdefdd6_b0p5_stratified_order_hold0p18_selclean_lrscale0p35_cw1p1x1p55x1p85_gm2p7_bad1p85_seed105_featwf_scalematch | ptb_val | 0.991274 | 0.990011 | 0.992629 | 0.994318 | 0.976562 | 3 | 6 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbankdefdd6_b0p5_stratified_order_hold0p18_selclean_lrscale0p35_cw1p1x1p55x1p85_gm2p7_bad1p85_seed105_featwf_scalematch | ptb_combo_stress_val | 0.966046 | 0.963711 | 1.000000 | 1.000000 | 0.897010 | 0 | 0 | 20 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbankdefdd6_b0p5_stratified_order_hold0p18_selclean_lrscale0p35_cw1p1x1p55x1p85_gm2p7_bad1p85_seed105_featwf_scalematch | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_val | 0.834357 | 0.853275 | 0.814271 | 0.799455 | 0.955253 | 556 | 340 | 41 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbankdefdd6_b0p5_stratified_order_hold0p18_selclean_lrscale0p35_cw1p1x1p55x1p85_gm2p7_bad1p85_seed105_featwf_scalematch | ptb_synthetic_test | 0.988724 | 0.987583 | 0.976987 | 0.995130 | 0.979253 | 11 | 6 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbankdefdd6_b0p5_stratified_order_hold0p18_selclean_lrscale0p35_cw1p1x1p55x1p85_gm2p7_bad1p85_seed105_featwf_scalematch | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_test | 0.828659 | 0.848341 | 0.816700 | 0.777115 | 0.955470 | 551 | 386 | 42 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbankdefdd6_b0p5_stratified_order_hold0p18_selclean_lrscale0p35_cw1p1x1p55x1p85_gm2p7_bad1p85_seed105_featwf_scalematch`: best_epoch=5, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=2078, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_ec935434f99e4f0b\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_ec935434f99e4f0b\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_defdd6_b0p5_stratified_order_hold0p18_sel_85f35435e9c5_metrics.csv`