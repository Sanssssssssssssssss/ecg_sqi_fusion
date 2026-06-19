# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p5_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1801 | ptb_val | 0.991856 | 0.990568 | 0.990172 | 0.996212 | 0.976562 | 4 | 4 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p5_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1801 | but_clean_margin_ge_5s_drop_outlier_val | 0.592481 | 0.317282 | 0.570048 | 0.930233 | 0.000000 | 267 | 3 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p5_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1801 | ptb_synthetic_test | 0.990774 | 0.989509 | 0.976987 | 0.998377 | 0.979253 | 11 | 2 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p5_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1801 | but_clean_margin_ge_5s_drop_outlier_test | 0.875899 | 0.710338 | 0.892430 | 0.904670 | 0.228814 | 108 | 198 | 91 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p5_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1801`: best_epoch=5, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=725, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_63a3dc1a185fcab1\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_d592fe_b0p5_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1801_metrics.csv`