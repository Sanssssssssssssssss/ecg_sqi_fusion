# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p35_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed71_featwf_scalematch | ptb_val | 0.962187 | 0.953232 | 0.923833 | 0.973485 | 0.976562 | 26 | 0 | 6 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p35_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed71_featwf_scalematch | ptb_combo_stress_val | 0.870040 | 0.870660 | 1.000000 | 0.995556 | 0.674242 | 0 | 0 | 96 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p35_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed71_featwf_scalematch | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_val | 0.779768 | 0.784940 | 0.735912 | 0.745504 | 0.968872 | 712 | 224 | 32 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p35_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed71_featwf_scalematch | ptb_synthetic_test | 0.943619 | 0.922514 | 0.905858 | 0.948864 | 0.991701 | 29 | 1 | 2 | 0 |
| ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p35_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed71_featwf_scalematch | but_clean_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_test | 0.779364 | 0.785596 | 0.740852 | 0.738069 | 0.965150 | 701 | 260 | 36 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_keep_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p35_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed71_featwf_scalematch`: best_epoch=5, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1607, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_4a4892b7d7cd23fb\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_4a4892b7d7cd23fb\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_2f20ac_b0p35_stratified_order_hold0p18_se_70c5eef003d9_metrics.csv`