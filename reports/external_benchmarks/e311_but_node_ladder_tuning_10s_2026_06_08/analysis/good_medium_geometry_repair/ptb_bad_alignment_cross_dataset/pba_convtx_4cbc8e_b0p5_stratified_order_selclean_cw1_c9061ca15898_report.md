# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_selclean_cw1p16x1p55x2p05_gm2p2_bad2p0_seed1875_scalematch | ptb_val | 0.945899 | 0.933753 | 0.990172 | 0.916667 | 0.996094 | 4 | 3 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_selclean_cw1p16x1p55x2p05_gm2p2_bad2p0_seed1875_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.923291 | 0.929489 | 0.949084 | 0.826923 | 1.000000 | 114 | 212 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_selclean_cw1p16x1p55x2p05_gm2p2_bad2p0_seed1875_scalematch | ptb_synthetic_test | 0.990261 | 0.989704 | 0.968619 | 0.998377 | 0.991701 | 15 | 1 | 2 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_selclean_cw1p16x1p55x2p05_gm2p2_bad2p0_seed1875_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.920760 | 0.927854 | 0.943850 | 0.827888 | 1.000000 | 126 | 215 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_selclean_cw1p16x1p55x2p05_gm2p2_bad2p0_seed1875_scalematch`: best_epoch=3, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1738, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_933f7da07d6a3b58\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_933f7da07d6a3b58\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_4cbc8e_b0p5_stratified_order_selclean_cw1_c9061ca15898_metrics.csv`