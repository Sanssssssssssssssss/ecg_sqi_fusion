# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p35_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed72_featwf_scalematch | ptb_val | 0.930192 | 0.919048 | 0.850123 | 0.946023 | 0.992188 | 60 | 0 | 2 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p35_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed72_featwf_scalematch | ptb_combo_stress_val | 0.806548 | 0.805346 | 0.913580 | 0.982222 | 0.563131 | 14 | 0 | 138 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p35_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed72_featwf_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.853092 | 0.870700 | 0.760161 | 0.923878 | 1.000000 | 536 | 54 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p35_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed72_featwf_scalematch | ptb_synthetic_test | 0.907740 | 0.888117 | 0.765690 | 0.946429 | 0.991701 | 106 | 1 | 2 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p35_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed72_featwf_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.851946 | 0.869036 | 0.764260 | 0.912351 | 1.000000 | 528 | 65 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p35_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed72_featwf_scalematch`: best_epoch=3, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1607, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_340592d41c3a5fa4\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_340592d41c3a5fa4\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_2f20ac_b0p35_stratified_order_hold0p18_se_c06119e37122_metrics.csv`