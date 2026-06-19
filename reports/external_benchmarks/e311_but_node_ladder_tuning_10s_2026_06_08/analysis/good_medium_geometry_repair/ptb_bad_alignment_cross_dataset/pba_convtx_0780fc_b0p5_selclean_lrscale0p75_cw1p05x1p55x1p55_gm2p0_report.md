# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_dualview_convtx_hier_p00_s0p0_distbank0780fc_b0p5_selclean_lrscale0p75_cw1p05x1p55x1p55_gm2p0 | ptb_val | 0.985457 | 0.984401 | 0.987715 | 0.986742 | 0.976562 | 5 | 14 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_dualview_convtx_hier_p00_s0p0_distbank0780fc_b0p5_selclean_lrscale0p75_cw1p05x1p55x1p55_gm2p0 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_val | 0.888350 | 0.890084 | 0.988894 | 0.635930 | 1.000000 | 25 | 458 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_dualview_convtx_hier_p00_s0p0_distbank0780fc_b0p5_selclean_lrscale0p75_cw1p05x1p55x1p55_gm2p0 | ptb_synthetic_test | 0.980010 | 0.979566 | 0.983264 | 0.978896 | 0.979253 | 8 | 26 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_dualview_convtx_hier_p00_s0p0_distbank0780fc_b0p5_selclean_lrscale0p75_cw1p05x1p55x1p55_gm2p0 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_test | 0.881567 | 0.882918 | 0.985385 | 0.621835 | 0.996333 | 33 | 478 | 3 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_dualview_convtx_hier_p00_s0p0_distbank0780fc_b0p5_selclean_lrscale0p75_cw1p05x1p55x1p55_gm2p0`: best_epoch=3, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=486, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_bd4ee3aa960b818f\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_0780fc_b0p5_selclean_lrscale0p75_cw1p05x1p55x1p55_gm2p0_metrics.csv`