# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed707 | ptb_val | 0.964514 | 0.957866 | 0.936118 | 0.968750 | 0.992188 | 26 | 0 | 2 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed707 | but_clean_margin_ge_5s_drop_outlier_val | 0.873684 | 0.474203 | 0.869565 | 0.953488 | 0.000000 | 81 | 2 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed707 | ptb_synthetic_test | 0.968221 | 0.961464 | 0.928870 | 0.978896 | 0.991701 | 34 | 2 | 2 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed707 | but_clean_margin_ge_5s_drop_outlier_test | 0.929040 | 0.934530 | 0.939243 | 0.920077 | 1.000000 | 61 | 157 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed707`: best_epoch=5, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=640, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_0af4de989b083919\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed707_metrics.csv`