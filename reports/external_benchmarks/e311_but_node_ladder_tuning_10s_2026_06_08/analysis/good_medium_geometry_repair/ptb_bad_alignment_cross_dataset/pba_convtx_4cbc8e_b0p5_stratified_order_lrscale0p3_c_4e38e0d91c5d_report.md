# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p3_cw1p20x1p60x2p05_gm2p5_bad2p0_seed1883_scalematch | ptb_val | 0.974404 | 0.967862 | 0.975430 | 0.970644 | 0.988281 | 10 | 0 | 3 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p3_cw1p20x1p60x2p05_gm2p5_bad2p0_seed1883_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.788703 | 0.820362 | 0.630192 | 0.980769 | 0.930061 | 828 | 24 | 57 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p3_cw1p20x1p60x2p05_gm2p5_bad2p0_seed1883_scalematch | ptb_synthetic_test | 0.980010 | 0.979699 | 0.926778 | 0.999188 | 0.987552 | 35 | 1 | 3 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p3_cw1p20x1p60x2p05_gm2p5_bad2p0_seed1883_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.784754 | 0.818141 | 0.618538 | 0.980876 | 0.940024 | 856 | 24 | 49 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p3_cw1p20x1p60x2p05_gm2p5_bad2p0_seed1883_scalematch`: best_epoch=7, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1738, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_a1df69b917d20974\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_a1df69b917d20974\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_4cbc8e_b0p5_stratified_order_lrscale0p3_c_4e38e0d91c5d_metrics.csv`