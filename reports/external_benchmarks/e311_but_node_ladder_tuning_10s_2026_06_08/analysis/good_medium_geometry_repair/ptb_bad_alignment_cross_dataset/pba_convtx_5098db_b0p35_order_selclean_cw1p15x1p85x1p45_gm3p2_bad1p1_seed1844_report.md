# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p35_order_selclean_cw1p15x1p85x1p45_gm3p2_bad1p1_seed1844 | ptb_val | 0.990692 | 0.989162 | 0.977887 | 0.999053 | 0.976562 | 9 | 0 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p35_order_selclean_cw1p15x1p85x1p45_gm3p2_bad1p1_seed1844 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.890981 | 0.888435 | 0.968289 | 0.743590 | 0.904294 | 71 | 320 | 78 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p35_order_selclean_cw1p15x1p85x1p45_gm3p2_bad1p1_seed1844 | ptb_synthetic_test | 0.984623 | 0.983577 | 0.951883 | 0.998377 | 0.979253 | 23 | 2 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p35_order_selclean_cw1p15x1p85x1p45_gm3p2_bad1p1_seed1844 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.892956 | 0.891827 | 0.970143 | 0.736255 | 0.921665 | 67 | 331 | 64 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbank5098db_b0p35_order_selclean_cw1p15x1p85x1p45_gm3p2_bad1p1_seed1844`: best_epoch=2, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=507, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_0412f4675f33b2f8\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_5098db_b0p35_order_selclean_cw1p15x1p85x1p45_gm3p2_bad1p1_seed1844_metrics.csv`