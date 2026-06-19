# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p65_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p58x2p05_gm2p35_bad2p05_seed31_scalematch | ptb_val | 0.984293 | 0.983862 | 0.992629 | 0.981061 | 0.984375 | 3 | 20 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p65_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p58x2p05_gm2p35_bad2p05_seed31_scalematch | ptb_combo_stress_val | 0.995427 | 0.995551 | 1.000000 | 0.995708 | 0.992337 | 0 | 1 | 2 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p65_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p58x2p05_gm2p35_bad2p05_seed31_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.908182 | 0.912510 | 0.963823 | 0.760417 | 0.981595 | 81 | 299 | 15 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p65_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p58x2p05_gm2p35_bad2p05_seed31_scalematch | ptb_synthetic_test | 0.971809 | 0.972569 | 0.960251 | 0.973214 | 0.987552 | 19 | 33 | 3 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p65_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p58x2p05_gm2p35_bad2p05_seed31_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.907322 | 0.911526 | 0.964349 | 0.758566 | 0.979192 | 80 | 303 | 17 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p65_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p14x1p58x2p05_gm2p35_bad2p05_seed31_scalematch`: best_epoch=4, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1944, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_b7e8aba4a1111b9b\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_b7e8aba4a1111b9b\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_60f7ad_b0p65_stratified_order_hold0p18_se_d57d9dd46e79_metrics.csv`