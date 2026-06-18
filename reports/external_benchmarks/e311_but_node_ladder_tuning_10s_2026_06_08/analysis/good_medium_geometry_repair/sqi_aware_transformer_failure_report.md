# SQI-Aware Transformer Failure Analysis

This analysis is report-only. It uses synthetic-selected checkpoints and original BUT only for diagnostics.

## What Broke

- `sqiquery_gm_focus_raw` keeps good/medium mostly reasonable but still misses almost all bad outlier stress.
- `sqiquery_badstress_hardneg_raw` learns bad stress, but it pays for it by converting a large fraction of original good/medium into bad.
- `sqiquery_multiview_robust3` does not solve the tradeoff; robust derivative/baseline channels make original good recall worse.

## Original Test Summary

| Candidate | Acc | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---:|---:|---:|---:|---:|---:|---:|
| sqiquery_gm_focus_raw | 0.793441 | 0.667582 | 0.957298 | 0.143552 | 1210 | 188 | 289 |
| sqiquery_badstress_hardneg_raw | 0.272974 | 0.166484 | 0.314957 | 0.763990 | 274 | 58 | 59 |
| sqiquery_multiview_robust3 | 0.558334 | 0.397527 | 0.711026 | 0.338200 | 571 | 719 | 124 |

## Weak Teacher Targets

| Feature | Mean Corr | Mean MAE(z) |
|---|---:|---:|
| qrs_visibility | 0.1633 | 0.8935 |
| sqi_basSQI | 0.1905 | 0.8999 |
| detector_agreement | 0.2370 | 0.6851 |
| region_confidence | 0.3109 | 0.7936 |
| baseline_step | 0.3160 | 0.8250 |
| boundary_confidence | 0.3295 | 0.6357 |
| low_amp_ratio | 0.3657 | 0.6670 |
| sqi_sSQI | 0.4051 | 0.8259 |

## Error Counts

- Badstress false-bad on nonbad original-test rows: `5734`.
- Badstress rescued bad outlier rows: `195`.
- GM-focus good->medium rows: `1210`.
- GM-focus recalled bad outlier rows: `0`.

## Figures

![Original test confusion heatmaps](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\sqi_aware_transformer_failure\sqi_aware_original_test_confusions.png)
![Key teacher recovery](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\sqi_aware_transformer_failure\sqi_aware_key_teacher_recovery.png)
![Failure waveform panels](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\sqi_aware_transformer_failure\sqi_aware_failure_waveform_panels.png)

## Next Hypothesis

A single 3-class Transformer softmax is not the right next test. The next experiment should separate representation learning from decision logic: keep a good/medium-focused waveform student, add a high-specificity bad-stress head, and train the bad head with explicit nonbad stress hard negatives plus a synthetic-only nonbad-floor selection rule. If that still fails, the missing piece is bad-stress data generation rather than architecture.
