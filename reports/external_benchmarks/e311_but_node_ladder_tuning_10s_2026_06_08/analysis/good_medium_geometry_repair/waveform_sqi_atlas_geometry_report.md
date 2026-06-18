# Waveform SQI Atlas Geometry

Atlas is built only from synthetic train SQI-plus features and labels. Original BUT is report-only.

## Best Original Test Report-Only

| Candidate | Acc | Macro-F1 | Good R | Medium R | Bad R | Bad outlier R |
|---|---:|---:|---:|---:|---:|---:|
| sqi_geom_rf | 0.820809 | 0.589802 | 0.831868 | 0.883416 | 0.048662 | 0.020548 |
| geomonly_rf | 0.815383 | 0.589892 | 0.823901 | 0.878897 | 0.055961 | 0.054795 |
| sqi_geom_extra | 0.812316 | 0.581977 | 0.798077 | 0.895165 | 0.046229 | 0.023973 |
| geomonly_extra | 0.809484 | 0.585604 | 0.792308 | 0.893583 | 0.055961 | 0.044521 |
| sqi_geom_mlp | 0.807007 | 0.557701 | 0.815110 | 0.874153 | 0.012165 | 0.017123 |
| geomonly_logreg | 0.802407 | 0.616928 | 0.787088 | 0.877316 | 0.131387 | 0.075342 |
| sqi_geom_logreg | 0.801227 | 0.674882 | 0.819780 | 0.835969 | 0.262774 | 0.003425 |
| geomonly_mlp | 0.789548 | 0.548430 | 0.784615 | 0.865341 | 0.017032 | 0.023973 |
| geomonly_histgb_lite | 0.522119 | 0.228681 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| geomonly_histgb | 0.522119 | 0.228681 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| sqi_geom_histgb | 0.522119 | 0.228681 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| sqi_geom_histgb_lite | 0.522119 | 0.228681 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |

## Synthetic Val/Test

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R |
|---|---|---:|---:|---:|---:|---:|
| geomonly_extra | synthetic_test | 0.983598 | 0.981960 | 0.956067 | 0.995130 | 0.979253 |
| geomonly_extra | synthetic_val | 0.979058 | 0.975426 | 0.977887 | 0.978220 | 0.984375 |
| geomonly_histgb | synthetic_test | 0.631471 | 0.258037 | 0.000000 | 1.000000 | 0.000000 |
| geomonly_histgb | synthetic_val | 0.614311 | 0.253694 | 0.000000 | 1.000000 | 0.000000 |
| geomonly_histgb_lite | synthetic_test | 0.631471 | 0.258037 | 0.000000 | 1.000000 | 0.000000 |
| geomonly_histgb_lite | synthetic_val | 0.614311 | 0.253694 | 0.000000 | 1.000000 | 0.000000 |
| geomonly_logreg | synthetic_test | 0.983598 | 0.982300 | 0.953975 | 0.995130 | 0.983402 |
| geomonly_logreg | synthetic_val | 0.970332 | 0.964560 | 0.977887 | 0.961174 | 0.996094 |
| geomonly_mlp | synthetic_test | 0.983598 | 0.981929 | 0.951883 | 0.996753 | 0.979253 |
| geomonly_mlp | synthetic_val | 0.979639 | 0.976486 | 0.982801 | 0.976326 | 0.988281 |
| geomonly_rf | synthetic_test | 0.982573 | 0.980337 | 0.958159 | 0.992695 | 0.979253 |
| geomonly_rf | synthetic_val | 0.977894 | 0.973489 | 0.985258 | 0.973485 | 0.984375 |
| sqi_geom_extra | synthetic_test | 0.984111 | 0.982456 | 0.958159 | 0.995130 | 0.979253 |
| sqi_geom_extra | synthetic_val | 0.979639 | 0.976000 | 0.980344 | 0.978220 | 0.984375 |
| sqi_geom_histgb | synthetic_test | 0.631471 | 0.258037 | 0.000000 | 1.000000 | 0.000000 |
| sqi_geom_histgb | synthetic_val | 0.614311 | 0.253694 | 0.000000 | 1.000000 | 0.000000 |
| sqi_geom_histgb_lite | synthetic_test | 0.631471 | 0.258037 | 0.000000 | 1.000000 | 0.000000 |
| sqi_geom_histgb_lite | synthetic_val | 0.614311 | 0.253694 | 0.000000 | 1.000000 | 0.000000 |
| sqi_geom_logreg | synthetic_test | 0.992312 | 0.992023 | 0.979079 | 0.997565 | 0.991701 |
| sqi_geom_logreg | synthetic_val | 0.974985 | 0.969839 | 0.985258 | 0.965909 | 0.996094 |
| sqi_geom_mlp | synthetic_test | 0.991799 | 0.990837 | 0.979079 | 0.998377 | 0.983402 |
| sqi_geom_mlp | synthetic_val | 0.991274 | 0.990038 | 0.992629 | 0.992424 | 0.984375 |
| sqi_geom_rf | synthetic_test | 0.985648 | 0.983955 | 0.966527 | 0.994318 | 0.979253 |
| sqi_geom_rf | synthetic_val | 0.980803 | 0.977363 | 0.985258 | 0.978220 | 0.984375 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_sqi_atlas_geometry_metrics.csv`
- Importance CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_sqi_atlas_geometry_importance.csv`
- Schema JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_sqi_atlas_geometry_schema.json`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_sqi_atlas_geometry_summary.json`
