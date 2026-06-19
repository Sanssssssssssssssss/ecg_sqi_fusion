# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p35_cw1p22x1p58x2p05_gm2p4_bad2p0_seed1882_scalematch | ptb_val | 0.977312 | 0.970484 | 0.992629 | 0.969697 | 0.984375 | 3 | 1 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p35_cw1p22x1p58x2p05_gm2p4_bad2p0_seed1882_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.817992 | 0.849330 | 0.666369 | 0.971154 | 1.000000 | 747 | 36 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p35_cw1p22x1p58x2p05_gm2p4_bad2p0_seed1882_scalematch | ptb_synthetic_test | 0.987186 | 0.986416 | 0.962343 | 0.997565 | 0.983402 | 18 | 3 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p35_cw1p22x1p58x2p05_gm2p4_bad2p0_seed1882_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.820899 | 0.851824 | 0.668004 | 0.977689 | 1.000000 | 745 | 28 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p35_cw1p22x1p58x2p05_gm2p4_bad2p0_seed1882_scalematch`: best_epoch=8, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1738, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_20d55f697c687287\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_20d55f697c687287\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_4cbc8e_b0p5_stratified_order_lrscale0p35__80dae32203da_metrics.csv`