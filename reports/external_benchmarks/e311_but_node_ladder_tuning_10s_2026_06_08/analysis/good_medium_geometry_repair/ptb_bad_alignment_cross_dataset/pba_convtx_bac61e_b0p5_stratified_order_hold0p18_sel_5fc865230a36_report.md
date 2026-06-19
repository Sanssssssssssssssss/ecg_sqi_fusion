# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbankbac61e_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p16x1p5x2p05_gm2p15_bad2p05_seed51_scalematch | ptb_val | 0.979058 | 0.973695 | 0.992629 | 0.971591 | 0.988281 | 3 | 6 | 3 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbankbac61e_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p16x1p5x2p05_gm2p15_bad2p05_seed51_scalematch | ptb_combo_stress_val | 0.998855 | 0.998990 | 1.000000 | 1.000000 | 0.996169 | 0 | 0 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbankbac61e_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p16x1p5x2p05_gm2p15_bad2p05_seed51_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.920270 | 0.926438 | 0.952657 | 0.810096 | 1.000000 | 106 | 234 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbankbac61e_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p16x1p5x2p05_gm2p15_bad2p05_seed51_scalematch | ptb_synthetic_test | 0.984623 | 0.982423 | 0.976987 | 0.987013 | 0.987552 | 11 | 10 | 3 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbankbac61e_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p16x1p5x2p05_gm2p15_bad2p05_seed51_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.915431 | 0.921419 | 0.955437 | 0.788845 | 1.000000 | 100 | 261 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbankbac61e_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p16x1p5x2p05_gm2p15_bad2p05_seed51_scalematch`: best_epoch=5, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1988, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_7066458aa7962445\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_7066458aa7962445\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_bac61e_b0p5_stratified_order_hold0p18_sel_5fc865230a36_metrics.csv`