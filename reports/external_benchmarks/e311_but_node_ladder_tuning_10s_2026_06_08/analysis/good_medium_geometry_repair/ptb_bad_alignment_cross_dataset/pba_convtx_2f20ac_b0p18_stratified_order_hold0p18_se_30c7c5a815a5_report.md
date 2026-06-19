# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p18_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed98_featwf_scalematch | ptb_val | 0.981966 | 0.978757 | 0.963145 | 0.990530 | 0.976562 | 15 | 1 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p18_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed98_featwf_scalematch | ptb_combo_stress_val | 0.802579 | 0.797134 | 1.000000 | 0.988889 | 0.510101 | 0 | 0 | 143 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p18_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed98_featwf_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.884240 | 0.889560 | 0.865118 | 0.900641 | 0.911656 | 301 | 113 | 72 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p18_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed98_featwf_scalematch | ptb_synthetic_test | 0.972322 | 0.967337 | 0.935146 | 0.984578 | 0.983402 | 31 | 4 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p18_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed98_featwf_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.882067 | 0.886240 | 0.864082 | 0.902789 | 0.899633 | 304 | 108 | 82 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p18_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed98_featwf_scalematch`: best_epoch=5, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=827, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_4c0e23fc3183d844\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_4c0e23fc3183d844\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_2f20ac_b0p18_stratified_order_hold0p18_se_30c7c5a815a5_metrics.csv`