# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5_selclean_lrscale0p75_cw1p05x1p55x1p55_gm2p0 | ptb_val | 0.988947 | 0.986582 | 0.992629 | 0.989583 | 0.980469 | 3 | 5 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5_selclean_lrscale0p75_cw1p05x1p55x1p55_gm2p0 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_val | 0.915858 | 0.920477 | 0.980009 | 0.746423 | 1.000000 | 45 | 319 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5_selclean_lrscale0p75_cw1p05x1p55x1p55_gm2p0 | ptb_synthetic_test | 0.992824 | 0.991470 | 0.987448 | 0.997565 | 0.979253 | 6 | 3 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5_selclean_lrscale0p75_cw1p05x1p55x1p55_gm2p0 | but_clean_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_test | 0.909447 | 0.914263 | 0.975642 | 0.733386 | 0.998778 | 55 | 337 | 1 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_cv_seed20260619_fold0_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5_selclean_lrscale0p75_cw1p05x1p55x1p55_gm2p0`: best_epoch=5, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=972, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_70c78fe3d3ed317c\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_bb7b93_b0p5_selclean_lrscale0p75_cw1p05x1p55x1p55_gm2p0_metrics.csv`