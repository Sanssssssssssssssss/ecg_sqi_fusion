# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed707_featwf | ptb_val | 0.970913 | 0.963387 | 0.992629 | 0.959280 | 0.984375 | 3 | 7 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed707_featwf | but_clean_margin_ge_5s_drop_outlier_val | 0.978947 | 0.611317 | 0.987118 | 0.883721 | 0.000000 | 8 | 5 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed707_featwf | ptb_synthetic_test | 0.976422 | 0.976746 | 0.989540 | 0.969968 | 0.983402 | 5 | 37 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed707_featwf | but_clean_margin_ge_5s_drop_outlier_test | 0.713348 | 0.800974 | 0.993028 | 0.561868 | 1.000000 | 7 | 910 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed707_featwf`: best_epoch=3, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=640, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_160344f6fd55c9a5\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed707_featwf_metrics.csv`