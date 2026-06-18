# Waveform Teacher Recovery Probe

Analysis-only probe. Trains on synthetic teacher targets and reports synthetic-test feature recovery.

## Key Teacher Recovery

| Candidate | Feature | MAE(z) | Corr |
|---|---|---:|---:|
| patchstat_teacher | pc2 | 1.0207 | 0.0269 |
| patchstat_teacher | qrs_visibility | 0.9629 | 0.1612 |
| patchstat_teacher | sqi_basSQI | 0.9063 | 0.1817 |
| patchstat_teacher | detector_agreement | 0.6826 | 0.2421 |
| patchstat_teacher | region_confidence | 0.7785 | 0.2963 |
| patchstat_teacher | baseline_step | 0.8426 | 0.2998 |
| patchstat_teacher | boundary_confidence | 0.6236 | 0.3281 |
| patchstat_teacher | pc3 | 0.7493 | 0.4032 |
| patchstat_teacher | sqi_sSQI | 0.8093 | 0.4065 |
| patchstat_teacher | knn_label_purity | 0.6698 | 0.4108 |
| stat_mlp_teacher | pc2 | 0.9942 | 0.0436 |
| stat_mlp_teacher | qrs_visibility | 0.8698 | 0.1883 |
| stat_mlp_teacher | sqi_basSQI | 0.8933 | 0.1940 |
| stat_mlp_teacher | detector_agreement | 0.6888 | 0.2397 |
| stat_mlp_teacher | region_confidence | 0.7907 | 0.2870 |
| stat_mlp_teacher | boundary_confidence | 0.6395 | 0.3045 |
| stat_mlp_teacher | baseline_step | 0.8203 | 0.3139 |
| stat_mlp_teacher | pc3 | 0.7653 | 0.3476 |
| stat_mlp_teacher | knn_label_purity | 0.6842 | 0.3792 |
| stat_mlp_teacher | sqi_sSQI | 0.8205 | 0.3877 |

## Readout

- If `stat_mlp_teacher` recovers a feature but `patchstat_teacher` does not, the signal exists in waveform-derived statistics but the Transformer tokenization/loss is failing.
- If neither recovers a feature, that teacher target is likely a dataset-geometry target rather than a directly learnable waveform invariant.
- Full CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_teacher_recovery_probe.csv`
