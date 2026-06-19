# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank04f776_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_featwf | ptb_val | 0.979058 | 0.976788 | 0.931204 | 0.998106 | 0.976562 | 27 | 0 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank04f776_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_featwf | ptb_combo_stress_val | 0.858503 | 0.847304 | 0.992000 | 0.991968 | 0.576271 | 2 | 0 | 86 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank04f776_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_featwf | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_val | 0.830774 | 0.849647 | 0.800934 | 0.807084 | 0.960117 | 584 | 321 | 41 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank04f776_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_featwf | ptb_synthetic_test | 0.965146 | 0.955527 | 0.916318 | 0.981331 | 0.979253 | 34 | 3 | 5 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank04f776_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_featwf | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_test | 0.828999 | 0.847393 | 0.808716 | 0.787961 | 0.961278 | 563 | 355 | 39 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_stressaux0p2_artaux0p12_artnbbad0p08_distbank04f776_b1p0_stratified_order_hold0p1_selclean_lrscale0p6_cw1x1p35x1p5_gm4p0_bad1p6_featwf`: best_epoch=3, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=1291, matched_bank_added=6621, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_840f7312abb777a7\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_840f7312abb777a7\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_04f776_b1p0_stratified_order_hold0p1_stre_b0b37f382dcf_metrics.csv`