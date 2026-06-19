# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p35_order_cw1p2x1p55x2_gm2p2_bad2p0_seed1840 | ptb_val | 0.994183 | 0.992841 | 1.000000 | 0.996212 | 0.976562 | 0 | 4 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p35_order_cw1p2x1p55x2_gm2p2_bad2p0_seed1840 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.691306 | 0.476054 | 0.991514 | 0.604167 | 0.000000 | 19 | 494 | 815 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p35_order_cw1p2x1p55x2_gm2p2_bad2p0_seed1840 | ptb_synthetic_test | 0.984623 | 0.983863 | 0.991632 | 0.982955 | 0.979253 | 4 | 21 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p35_order_cw1p2x1p55x2_gm2p2_bad2p0_seed1840 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.691613 | 0.476913 | 0.989750 | 0.608765 | 0.000000 | 23 | 491 | 817 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p35_order_cw1p2x1p55x2_gm2p2_bad2p0_seed1840`: best_epoch=4, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=507, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_0557808dda93b83e\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_0557808dda93b83e\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_5098db_b0p35_order_cw1p2x1p55x2_gm2p2_bad2p0_seed1840_metrics.csv`