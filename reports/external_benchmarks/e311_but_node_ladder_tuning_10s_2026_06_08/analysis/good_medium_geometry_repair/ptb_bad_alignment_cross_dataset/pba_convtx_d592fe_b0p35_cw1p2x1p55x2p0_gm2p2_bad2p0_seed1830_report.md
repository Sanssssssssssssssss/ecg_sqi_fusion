# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1830 | ptb_val | 0.995346 | 0.993969 | 0.995086 | 1.000000 | 0.976562 | 2 | 0 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1830 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_val | 0.739656 | 0.533407 | 0.968736 | 0.802885 | 0.013497 | 70 | 246 | 804 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1830 | ptb_synthetic_test | 0.991799 | 0.990866 | 0.987448 | 0.995130 | 0.983402 | 6 | 6 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1830 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_test | 0.743281 | 0.534613 | 0.974599 | 0.806375 | 0.011016 | 57 | 243 | 808 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold2_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1830`: best_epoch=4, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=507, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_b1f137d117c63796\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_d592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1830_metrics.csv`