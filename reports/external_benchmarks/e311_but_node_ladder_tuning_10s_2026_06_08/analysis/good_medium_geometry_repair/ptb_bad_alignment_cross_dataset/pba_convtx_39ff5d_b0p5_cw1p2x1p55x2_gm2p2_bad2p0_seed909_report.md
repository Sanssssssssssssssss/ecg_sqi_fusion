# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed909 | ptb_val | 0.942408 | 0.930775 | 0.992629 | 0.914773 | 0.976562 | 3 | 22 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed909 | but_clean_margin_ge_5s_drop_outlier_val | 0.972932 | 0.720984 | 0.990338 | 0.720930 | 1.000000 | 6 | 9 | 0 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed909 | ptb_synthetic_test | 0.952845 | 0.942659 | 0.989540 | 0.933442 | 0.979253 | 5 | 43 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed909 | but_clean_margin_ge_5s_drop_outlier_test | 0.614567 | 0.723529 | 0.988048 | 0.412133 | 1.000000 | 12 | 1214 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed909`: best_epoch=2, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=640, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_660903c37603744a\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_39ff5d_b0p5_cw1p2x1p55x2_gm2p2_bad2p0_seed909_metrics.csv`