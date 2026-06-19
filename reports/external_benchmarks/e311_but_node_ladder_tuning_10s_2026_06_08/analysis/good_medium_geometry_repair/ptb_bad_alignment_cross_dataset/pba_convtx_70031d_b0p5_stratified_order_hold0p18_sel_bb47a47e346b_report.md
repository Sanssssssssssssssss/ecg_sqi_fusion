# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank70031d_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed102_featwf_scalematch | ptb_val | 0.971495 | 0.967232 | 0.931204 | 0.985795 | 0.976562 | 28 | 1 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank70031d_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed102_featwf_scalematch | ptb_combo_stress_val | 0.922840 | 0.916821 | 1.000000 | 0.980000 | 0.816667 | 0 | 9 | 41 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank70031d_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed102_featwf_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.907485 | 0.915173 | 0.937919 | 0.792468 | 1.000000 | 139 | 258 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank70031d_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed102_featwf_scalematch | ptb_synthetic_test | 0.965659 | 0.965461 | 0.914226 | 0.982143 | 0.983402 | 41 | 21 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank70031d_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed102_featwf_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.904541 | 0.912459 | 0.934938 | 0.788048 | 1.000000 | 146 | 264 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank70031d_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed102_featwf_scalematch`: best_epoch=2, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=2214, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_0d1fcda49e1c1fc4\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_0d1fcda49e1c1fc4\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_70031d_b0p5_stratified_order_hold0p18_sel_bb47a47e346b_metrics.csv`