# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p5_stratified_order_hold0p18_selstressbal_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | ptb_val | 0.976149 | 0.975666 | 0.992629 | 0.969697 | 0.976562 | 3 | 32 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p5_stratified_order_hold0p18_selstressbal_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | ptb_combo_stress_val | 0.981707 | 0.982236 | 1.000000 | 0.982833 | 0.969349 | 0 | 4 | 8 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p5_stratified_order_hold0p18_selstressbal_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.705021 | 0.492381 | 0.977222 | 0.677083 | 0.000000 | 51 | 403 | 815 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p5_stratified_order_hold0p18_selstressbal_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | ptb_synthetic_test | 0.972834 | 0.973035 | 0.981172 | 0.968344 | 0.979253 | 9 | 39 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p5_stratified_order_hold0p18_selstressbal_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.708295 | 0.494582 | 0.983066 | 0.678088 | 0.000000 | 38 | 404 | 817 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p5_stratified_order_hold0p18_selstressbal_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch`: best_epoch=5, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1495, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_40f8a46aa4db779e\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_40f8a46aa4db779e\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_60f7ad_b0p5_stratified_order_hold0p18_sel_cdadb4af0a8a_metrics.csv`