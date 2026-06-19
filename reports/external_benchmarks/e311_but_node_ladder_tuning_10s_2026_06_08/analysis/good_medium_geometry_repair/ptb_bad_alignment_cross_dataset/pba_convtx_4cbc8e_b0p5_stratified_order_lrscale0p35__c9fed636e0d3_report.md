# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p35_cw1p24x1p50x2p05_gm2p0_bad2p0_seed1881_scalematch | ptb_val | 0.990111 | 0.989161 | 0.992629 | 0.990530 | 0.984375 | 3 | 9 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p35_cw1p24x1p50x2p05_gm2p0_bad2p0_seed1881_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.921432 | 0.929825 | 0.922287 | 0.868590 | 1.000000 | 174 | 164 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p35_cw1p24x1p50x2p05_gm2p0_bad2p0_seed1881_scalematch | ptb_synthetic_test | 0.990774 | 0.990255 | 0.985356 | 0.993506 | 0.987552 | 7 | 8 | 3 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p35_cw1p24x1p50x2p05_gm2p0_bad2p0_seed1881_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.919601 | 0.928005 | 0.925134 | 0.857371 | 1.000000 | 168 | 179 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p35_cw1p24x1p50x2p05_gm2p0_bad2p0_seed1881_scalematch`: best_epoch=5, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1738, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_6d42dc2da6254a9f\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_6d42dc2da6254a9f\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_4cbc8e_b0p5_stratified_order_lrscale0p35__c9fed636e0d3_metrics.csv`