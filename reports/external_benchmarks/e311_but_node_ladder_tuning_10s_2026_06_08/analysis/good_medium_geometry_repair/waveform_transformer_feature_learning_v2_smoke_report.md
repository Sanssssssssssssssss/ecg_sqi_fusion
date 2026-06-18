# Transformer-Only Waveform Geometry Learning V2

Inference input is ECG waveform-derived channels only. The 47 SQI/geometry columns are training teacher targets and diagnostics only.

## Main Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m | Key Aux MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_qrs_filterbank_boundary | synthetic_test | 0.987186 | 0.985304 | 0.962343 | 0.997565 | 0.983402 | 16 | 3 | 0 | 4 | 0.7160 |
| hybrid_qrs_filterbank_boundary_badcal | synthetic_test | 0.985648 | 0.982170 | 0.956067 | 0.997565 | 0.983402 | 16 | 3 | 0 | 4 | nan |
| hybrid_qrs_filterbank_boundary | original_test_all_10s+ | 0.731155 | 0.572455 | 0.698352 | 0.801627 | 0.262774 | 786 | 455 | 0 | 148 | nan |
| hybrid_qrs_filterbank_boundary_badcal | original_test_all_10s+ | 0.708859 | 0.575350 | 0.672802 | 0.765703 | 0.416058 | 715 | 334 | 0 | 106 | nan |
| hybrid_qrs_filterbank_boundary | original_all_10s+ | 0.802191 | 0.814926 | 0.718183 | 0.872695 | 0.931315 | 4275 | 848 | 0 | 208 | nan |
| hybrid_qrs_filterbank_boundary_badcal | original_all_10s+ | 0.793027 | 0.799474 | 0.708033 | 0.854535 | 0.943425 | 4176 | 710 | 0 | 165 | nan |
| hybrid_qrs_filterbank_boundary | bad_core_nearboundary | 0.394958 | 0.188755 | 0.000000 | 0.000000 | 0.394958 | 0 | 0 | 0 | 72 | nan |
| hybrid_qrs_filterbank_boundary_badcal | bad_core_nearboundary | 0.605042 | 0.251309 | 0.000000 | 0.000000 | 0.605042 | 0 | 0 | 0 | 47 | nan |
| hybrid_qrs_filterbank_boundary | bad_outlier_stress | 0.208904 | 0.115203 | 0.000000 | 0.000000 | 0.208904 | 0 | 0 | 0 | 76 | nan |
| hybrid_qrs_filterbank_boundary_badcal | bad_outlier_stress | 0.339041 | 0.168798 | 0.000000 | 0.000000 | 0.339041 | 0 | 0 | 0 | 59 | nan |

## Missing-Feature Recovery Gates

| Candidate | Feature | Split | Corr | MAE(z) | Gate |
|---|---|---|---:|---:|---|
| hybrid_qrs_filterbank_boundary | baseline_step | synthetic_test | 0.3218 | 0.8124 | need >= 0.55 |
| hybrid_qrs_filterbank_boundary | detector_agreement | synthetic_test | 0.2426 | 0.6873 | need >= 0.45 |
| hybrid_qrs_filterbank_boundary | flatline_ratio | synthetic_test | 0.5434 | 0.7359 | need >= 0.60 |
| hybrid_qrs_filterbank_boundary | pca_margin | synthetic_test | 0.7458 | 0.3254 | pass |
| hybrid_qrs_filterbank_boundary | qrs_visibility | synthetic_test | 0.1600 | 0.8855 | need >= 0.45 |

## Weakest Key Teacher Targets

| Candidate | Feature | Corr | MAE(z) |
|---|---|---:|---:|
| hybrid_qrs_filterbank_boundary | pc2 | 0.0449 | 1.0011 |
| hybrid_qrs_filterbank_boundary | qrs_visibility | 0.1600 | 0.8855 |
| hybrid_qrs_filterbank_boundary | sqi_basSQI | 0.2040 | 0.8931 |
| hybrid_qrs_filterbank_boundary | detector_agreement | 0.2426 | 0.6873 |
| hybrid_qrs_filterbank_boundary | baseline_step | 0.3218 | 0.8124 |
| hybrid_qrs_filterbank_boundary | boundary_confidence | 0.3303 | 0.6255 |
| hybrid_qrs_filterbank_boundary | region_confidence | 0.3429 | 0.7777 |
| hybrid_qrs_filterbank_boundary | low_amp_ratio | 0.3699 | 0.6622 |

## Reference And Interpretation

- 47-feature upper bound original test: acc `0.963548`, good/medium/bad `0.956/0.973/0.927`.
- Original BUT is report-only. No original row is used for training, validation, threshold selection, or candidate selection.
- A candidate below 0.90 original acc is not just an accuracy failure; inspect its failed gates and waveform panels before changing architecture.

## Artifacts

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_feature_learning_v2_smoke_metrics.csv`
- Feature recovery CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_feature_learning_v2_smoke_feature_recovery.csv`
- `hybrid_qrs_filterbank_boundary` best_epoch=2 threshold=0.27 feature_gates=1/5 run_dir=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_feature_learning_v2\N17043_gm_probe\hybrid_qrs_filterbank_boundary`
