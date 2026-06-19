# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | ptb_val | 0.952880 | 0.942858 | 0.987715 | 0.930871 | 0.988281 | 5 | 15 | 3 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | ptb_combo_stress_val | 0.989329 | 0.989360 | 1.000000 | 0.982833 | 0.988506 | 0 | 3 | 3 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.910042 | 0.914109 | 0.952657 | 0.774840 | 1.000000 | 106 | 262 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | ptb_synthetic_test | 0.972322 | 0.969523 | 0.979079 | 0.964286 | 1.000000 | 10 | 30 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.910102 | 0.913467 | 0.954545 | 0.772112 | 1.000000 | 102 | 262 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p60_s1p0_distbank60f7ad_b0p5_stratified_order_hold0p18_selbadguard_lrscale0p35_cw1p18x1p52x2p05_gm2p15_bad2p05_scalematch`: best_epoch=4, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1495, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_96dcb5943310ae2e\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_96dcb5943310ae2e\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_60f7ad_b0p5_stratified_order_hold0p18_sel_6cf66e27335e_metrics.csv`