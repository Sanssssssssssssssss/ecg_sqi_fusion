# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1805 | ptb_val | 0.917394 | 0.912940 | 0.992629 | 0.874053 | 0.976562 | 3 | 94 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1805 | but_clean_margin_ge_5s_drop_outlier_val | 0.975940 | 0.595085 | 0.995169 | 0.720930 | 0.000000 | 3 | 9 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1805 | ptb_synthetic_test | 0.941056 | 0.945339 | 0.979079 | 0.918831 | 0.979253 | 10 | 100 | 5 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1805 | but_clean_margin_ge_5s_drop_outlier_test | 0.662394 | 0.747448 | 0.997012 | 0.481464 | 1.000000 | 3 | 1062 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1805`: best_epoch=1, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=507, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_6a363172defebad6\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_d592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1805_metrics.csv`