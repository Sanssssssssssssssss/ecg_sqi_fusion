# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank515b0a_b0p5_stratified_order_hold0p18_selstress_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | ptb_val | 0.987784 | 0.986570 | 0.975430 | 0.995265 | 0.976562 | 10 | 5 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank515b0a_b0p5_stratified_order_hold0p18_selstress_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | ptb_combo_stress_val | 0.972843 | 0.973181 | 0.962963 | 0.995708 | 0.956710 | 6 | 1 | 9 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank515b0a_b0p5_stratified_order_hold0p18_selstress_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.733612 | 0.531221 | 0.911568 | 0.890224 | 0.004908 | 198 | 137 | 811 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank515b0a_b0p5_stratified_order_hold0p18_selstress_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | ptb_synthetic_test | 0.982573 | 0.981707 | 0.958159 | 0.992695 | 0.979253 | 20 | 9 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank515b0a_b0p5_stratified_order_hold0p18_selstress_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.731233 | 0.530694 | 0.910428 | 0.882072 | 0.007344 | 201 | 148 | 811 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank515b0a_b0p5_stratified_order_hold0p18_selstress_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch`: best_epoch=3, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1426, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_6215969d1d5d85cc\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_6215969d1d5d85cc\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_515b0a_b0p5_stratified_order_hold0p18_sel_5eb89f421a90_metrics.csv`