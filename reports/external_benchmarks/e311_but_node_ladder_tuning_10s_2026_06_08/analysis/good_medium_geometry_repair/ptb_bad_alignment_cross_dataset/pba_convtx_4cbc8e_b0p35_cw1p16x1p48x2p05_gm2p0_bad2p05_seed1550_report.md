# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p35_cw1p16x1p48x2p05_gm2p0_bad2p05_seed1550 | ptb_val | 0.975567 | 0.968279 | 0.990172 | 0.969697 | 0.976562 | 4 | 1 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p35_cw1p16x1p48x2p05_gm2p0_bad2p05_seed1550 | but_clean_margin_ge_5s_drop_outlier_val | 0.975940 | 0.602032 | 0.987118 | 0.837209 | 0.000000 | 8 | 7 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p35_cw1p16x1p48x2p05_gm2p0_bad2p05_seed1550 | ptb_synthetic_test | 0.980010 | 0.979566 | 0.983264 | 0.978896 | 0.979253 | 8 | 26 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p35_cw1p16x1p48x2p05_gm2p0_bad2p05_seed1550 | but_clean_margin_ge_5s_drop_outlier_test | 0.633323 | 0.430313 | 0.993028 | 0.495426 | 0.000000 | 7 | 1048 | 118 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbank4cbc8e_b0p35_cw1p16x1p48x2p05_gm2p0_bad2p05_seed1550`: best_epoch=3, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=1217, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_39c14db6d75acba0\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_4cbc8e_b0p35_cw1p16x1p48x2p05_gm2p0_bad2p05_seed1550_metrics.csv`