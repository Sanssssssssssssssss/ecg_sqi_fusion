# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p5_cw1p22x1p52x2p05_gm2p1_bad2p0_seed1880_scalematch | ptb_val | 0.995928 | 0.995056 | 1.000000 | 0.996212 | 0.988281 | 0 | 3 | 3 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p5_cw1p22x1p52x2p05_gm2p1_bad2p0_seed1880_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.855416 | 0.878719 | 0.742296 | 0.963942 | 1.000000 | 577 | 43 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p5_cw1p22x1p52x2p05_gm2p1_bad2p0_seed1880_scalematch | ptb_synthetic_test | 0.991287 | 0.991042 | 0.974895 | 0.997565 | 0.991701 | 12 | 3 | 2 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p5_cw1p22x1p52x2p05_gm2p1_bad2p0_seed1880_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.860056 | 0.882814 | 0.749554 | 0.966534 | 1.000000 | 562 | 42 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p5_stratified_order_lrscale0p5_cw1p22x1p52x2p05_gm2p1_bad2p0_seed1880_scalematch`: best_epoch=8, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1738, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_c28e4a5a37482d67\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_c28e4a5a37482d67\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_4cbc8e_b0p5_stratified_order_lrscale0p5_c_affd232aa6fc_metrics.csv`