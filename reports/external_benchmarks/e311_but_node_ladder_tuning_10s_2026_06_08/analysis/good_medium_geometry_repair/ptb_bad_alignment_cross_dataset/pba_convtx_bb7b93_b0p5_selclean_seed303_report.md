# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5_selclean_seed303 | ptb_val | 0.977894 | 0.971298 | 0.992629 | 0.969697 | 0.988281 | 3 | 1 | 3 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5_selclean_seed303 | but_clean_margin_ge_5s_drop_outlier_val | 0.971429 | 0.593712 | 0.980676 | 0.860465 | 0.000000 | 12 | 6 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5_selclean_seed303 | ptb_synthetic_test | 0.988211 | 0.986836 | 0.981172 | 0.989448 | 0.995851 | 9 | 8 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5_selclean_seed303 | but_clean_margin_ge_5s_drop_outlier_test | 0.813379 | 0.862421 | 0.997012 | 0.714011 | 1.000000 | 3 | 590 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5_selclean_seed303`: best_epoch=3, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=972, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankbb7b93_b0p5_selclean_seed303\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_bb7b93_b0p5_selclean_seed303_metrics.csv`