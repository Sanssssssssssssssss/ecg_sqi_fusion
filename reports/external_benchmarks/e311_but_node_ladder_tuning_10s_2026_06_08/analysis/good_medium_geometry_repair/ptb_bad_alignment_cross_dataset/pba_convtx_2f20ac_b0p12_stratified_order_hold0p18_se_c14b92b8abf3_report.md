# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p12_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed92_featwf_scalematch | ptb_val | 0.981385 | 0.979425 | 0.943489 | 0.997159 | 0.976562 | 23 | 0 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p12_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed92_featwf_scalematch | ptb_combo_stress_val | 0.799603 | 0.801665 | 0.944444 | 0.982222 | 0.532828 | 9 | 0 | 152 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p12_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed92_featwf_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.813575 | 0.815886 | 0.767753 | 0.939904 | 0.746012 | 519 | 64 | 207 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p12_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed92_featwf_scalematch | ptb_synthetic_test | 0.971809 | 0.969043 | 0.910042 | 0.994318 | 0.979253 | 43 | 1 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p12_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed92_featwf_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.809082 | 0.806942 | 0.772727 | 0.938645 | 0.709914 | 510 | 62 | 237 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank2f20ac_b0p12_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p55x2p05_gm2p3_bad2p05_seed92_featwf_scalematch`: best_epoch=4, averaged_epochs=[], bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=551, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_3d96556a70f4214c\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_3d96556a70f4214c\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_2f20ac_b0p12_stratified_order_hold0p18_se_c14b92b8abf3_metrics.csv`