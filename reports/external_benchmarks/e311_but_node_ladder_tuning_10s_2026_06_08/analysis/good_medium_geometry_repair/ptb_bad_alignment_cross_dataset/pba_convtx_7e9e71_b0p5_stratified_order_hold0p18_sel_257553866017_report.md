# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank7e9e71_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p16x1p5x2p05_gm2p15_bad2p05_seed61_scalematch | ptb_val | 0.976149 | 0.969250 | 0.990172 | 0.967803 | 0.988281 | 4 | 1 | 3 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank7e9e71_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p16x1p5x2p05_gm2p15_bad2p05_seed61_scalematch | ptb_combo_stress_val | 0.998855 | 0.998990 | 1.000000 | 1.000000 | 0.996169 | 0 | 0 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank7e9e71_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p16x1p5x2p05_gm2p15_bad2p05_seed61_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.930265 | 0.937089 | 0.932559 | 0.880609 | 1.000000 | 151 | 145 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank7e9e71_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p16x1p5x2p05_gm2p15_bad2p05_seed61_scalematch | ptb_synthetic_test | 0.982060 | 0.980237 | 0.953975 | 0.990260 | 0.995851 | 22 | 5 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank7e9e71_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p16x1p5x2p05_gm2p15_bad2p05_seed61_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.934893 | 0.941482 | 0.935829 | 0.890837 | 1.000000 | 144 | 134 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank7e9e71_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p16x1p5x2p05_gm2p15_bad2p05_seed61_scalematch`: best_epoch=5, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1988, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_7cd7446bccc0ba2c\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_7cd7446bccc0ba2c\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_7e9e71_b0p5_stratified_order_hold0p18_sel_257553866017_metrics.csv`