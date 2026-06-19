# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p0_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1860_scalematch | ptb_val | 0.995928 | 0.994536 | 1.000000 | 0.999053 | 0.976562 | 0 | 1 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p0_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1860_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.614133 | 0.455354 | 0.635105 | 0.977564 | 0.000000 | 817 | 28 | 815 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p0_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1860_scalematch | ptb_synthetic_test | 0.990774 | 0.989524 | 0.981172 | 0.996753 | 0.979253 | 9 | 4 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p0_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1860_scalematch | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.605422 | 0.449604 | 0.612745 | 0.986454 | 0.000000 | 869 | 17 | 817 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p0_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1860_scalematch`: best_epoch=4, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=0, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_b59ace2c75bc0ca2\ckpt_best.pt`, predictions=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_b59ace2c75bc0ca2\but_clean_val_test_predictions.csv`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_5098db_b0p0_order_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1860_scalematch_metrics.csv`