# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p65_stratified_order_hold0p18_selbadguard_lrscale0p3_cw1p05x1p75x2p1_gm3p0_bad2p1_seed41_scalematch | ptb_val | 0.987202 | 0.985809 | 0.985258 | 0.990530 | 0.976562 | 6 | 9 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p65_stratified_order_hold0p18_selbadguard_lrscale0p3_cw1p05x1p75x2p1_gm3p0_bad2p1_seed41_scalematch | ptb_combo_stress_val | 0.983232 | 0.982428 | 0.981481 | 0.982833 | 0.984674 | 3 | 4 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p65_stratified_order_hold0p18_selbadguard_lrscale0p3_cw1p05x1p75x2p1_gm3p0_bad2p1_seed41_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.839145 | 0.831387 | 0.982582 | 0.492788 | 0.975460 | 39 | 631 | 20 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p65_stratified_order_hold0p18_selbadguard_lrscale0p3_cw1p05x1p75x2p1_gm3p0_bad2p1_seed41_scalematch | ptb_synthetic_test | 0.974372 | 0.973980 | 0.974895 | 0.973214 | 0.979253 | 12 | 32 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p65_stratified_order_hold0p18_selbadguard_lrscale0p3_cw1p05x1p75x2p1_gm3p0_bad2p1_seed41_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.843605 | 0.837353 | 0.981729 | 0.509960 | 0.976744 | 41 | 612 | 19 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p65_stratified_order_hold0p18_selbadguard_lrscale0p3_cw1p05x1p75x2p1_gm3p0_bad2p1_seed41_scalematch`: best_epoch=2, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1944, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_9f714118b41775fb\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_9f714118b41775fb\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_60f7ad_b0p65_stratified_order_hold0p18_se_8d3c8e0cde01_metrics.csv`