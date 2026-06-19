# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p18_stratified_order_hold0p18_selclean_lrscale0p35_cw1p1x1p55x1p7_gm2p5_bad1p7_seed99_featwf_scalematch | ptb_val | 0.988947 | 0.987215 | 0.975430 | 0.997159 | 0.976562 | 10 | 1 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p18_stratified_order_hold0p18_selclean_lrscale0p35_cw1p1x1p55x1p7_gm2p5_bad1p7_seed99_featwf_scalematch | ptb_combo_stress_val | 0.774802 | 0.767768 | 1.000000 | 1.000000 | 0.426768 | 0 | 0 | 179 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p18_stratified_order_hold0p18_selclean_lrscale0p35_cw1p1x1p55x1p7_gm2p5_bad1p7_seed99_featwf_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.909112 | 0.917970 | 0.899955 | 0.876603 | 0.984049 | 223 | 153 | 13 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p18_stratified_order_hold0p18_selclean_lrscale0p35_cw1p1x1p55x1p7_gm2p5_bad1p7_seed99_featwf_scalematch | ptb_synthetic_test | 0.983086 | 0.980750 | 0.943515 | 0.998377 | 0.983402 | 24 | 2 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p18_stratified_order_hold0p18_selclean_lrscale0p35_cw1p1x1p55x1p7_gm2p5_bad1p7_seed99_featwf_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.903846 | 0.912778 | 0.897059 | 0.864542 | 0.982864 | 230 | 166 | 14 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p18_stratified_order_hold0p18_selclean_lrscale0p35_cw1p1x1p55x1p7_gm2p5_bad1p7_seed99_featwf_scalematch`: best_epoch=4, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=827, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_37e0e7f800af2a5c\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_37e0e7f800af2a5c\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_2f20ac_b0p18_stratified_order_hold0p18_se_229e53dc9f2a_metrics.csv`