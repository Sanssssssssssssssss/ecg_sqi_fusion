# PTB Bad-Alignment Cross-Dataset Check

Fixed 10s only. Training input remains PTB synthetic waveform-derived channels. The main BUT target is the curated clean protocol; legacy full-BUT buckets are emitted only when explicitly requested.

## Why This Exists

The clean BUT protocol removes low-confidence/outlier windows that can be contaminated by long-label/window extraction artifacts. This run tests whether PTB synthetic bad can align with the curated BUT bad core/near-boundary target without damaging good/medium behavior.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | nonbad->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1800 | ptb_val | 0.981966 | 0.978953 | 0.958231 | 0.992424 | 0.976562 | 17 | 0 | 6 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1800 | but_clean_margin_ge_5s_drop_outlier_val | 0.951880 | 0.561487 | 0.956522 | 0.906977 | 0.000000 | 27 | 3 | 1 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1800 | ptb_synthetic_test | 0.973860 | 0.973281 | 0.905858 | 0.998377 | 0.983402 | 45 | 2 | 4 | 0 |
| ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1800 | but_clean_margin_ge_5s_drop_outlier_test | 0.939043 | 0.950216 | 0.934263 | 0.937891 | 1.000000 | 66 | 127 | 0 | 0 |

## Candidate Summaries

- `ptb_badalign_margin_ge_5s_drop_outlier_dualview_convtx_hier_p00_s0p0_distbankd592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1800`: best_epoch=2, bad_aug_added=0, nonbad_guard_added=0, matched_bank_added=507, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\ptb_bad_alignment_cross_dataset\pba_run_e5cc3f052fb2f4b8\ckpt_best.pt`

## Decision Contract

- Keep only candidates that preserve PTB synthetic self-test near or above 0.95.
- Main cross-dataset evidence is the curated clean-BUT protocol selected by --policy.
- Legacy full-BUT diagnostics, if enabled, are historical stress checks and not a modeling target.

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset\pba_convtx_d592fe_b0p35_cw1p2x1p55x2p0_gm2p2_bad2p0_seed1800_metrics.csv`