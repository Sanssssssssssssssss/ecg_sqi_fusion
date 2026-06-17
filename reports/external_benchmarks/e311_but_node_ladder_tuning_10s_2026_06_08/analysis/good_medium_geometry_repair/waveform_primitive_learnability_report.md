# Waveform Primitive Hard-Feature Learnability

Diagnostic-only check: can waveform-derived primitive stats predict hard SQI/geometry targets?

## Classifier

- qrs_enhanced / synthetic_test: acc=0.994874, macro_f1=0.994475, good/medium/bad=0.994/0.996/0.992
- qrs_enhanced / original_test_all_10s+: acc=0.830714, macro_f1=0.855537, good/medium/bad=0.795/0.837/0.932
- qrs_enhanced_v2 / synthetic_test: acc=0.994874, macro_f1=0.994134, good/medium/bad=0.996/0.995/0.992
- qrs_enhanced_v2 / original_test_all_10s+: acc=0.827619, macro_f1=0.853095, good/medium/bad=0.789/0.837/0.932
- qrs_stress_v3 / synthetic_test: acc=0.993337, macro_f1=0.992682, good/medium/bad=0.992/0.994/0.992
- qrs_stress_v3 / original_test_all_10s+: acc=0.828286, macro_f1=0.853441, good/medium/bad=0.794/0.832/0.932

## Hard Feature Recovery

| bank | feature | corr | mae_z | top14 | hard |
|---|---|---:|---:|---|---|
| qrs_enhanced | pca_margin | 0.7499 | 0.3419 | True | False |
| qrs_enhanced | pc1 | 0.7385 | 0.3967 | True | False |
| qrs_enhanced | non_qrs_diff_p95 | 0.7246 | 0.4477 | True | False |
| qrs_enhanced | sqi_bSQI | 0.6290 | 0.4111 | True | False |
| qrs_enhanced | flatline_ratio | 0.5372 | 0.7093 | True | True |
| qrs_enhanced | template_corr | 0.4564 | 0.6524 | True | False |
| qrs_enhanced | amplitude_entropy | 0.4534 | 0.7134 | True | False |
| qrs_enhanced | qrs_band_ratio | 0.4323 | 0.8015 | True | False |
| qrs_enhanced | pc3 | 0.4222 | 0.7332 | False | True |
| qrs_enhanced | knn_label_purity | 0.4154 | 0.6894 | False | True |
| qrs_enhanced | low_amp_ratio | 0.3778 | 0.6621 | False | True |
| qrs_enhanced | boundary_confidence | 0.3494 | 0.6547 | False | True |
| qrs_enhanced | region_confidence | 0.3489 | 0.8085 | True | False |
| qrs_enhanced | baseline_step | 0.3338 | 0.7849 | True | True |
| qrs_enhanced | detector_agreement | 0.2487 | 0.7494 | True | True |
| qrs_enhanced | sqi_basSQI | 0.2043 | 0.8734 | True | True |
| qrs_enhanced | qrs_visibility | 0.1653 | 0.8429 | True | True |
| qrs_enhanced | pc2 | 0.1067 | 0.9623 | False | True |
| qrs_enhanced | mean_abs | 0.0440 | 0.8978 | True | False |
| qrs_enhanced_v2 | pca_margin | 0.7481 | 0.3440 | True | False |
| qrs_enhanced_v2 | pc1 | 0.7375 | 0.3969 | True | False |
| qrs_enhanced_v2 | non_qrs_diff_p95 | 0.7246 | 0.4474 | True | False |
| qrs_enhanced_v2 | sqi_bSQI | 0.6304 | 0.4097 | True | False |
| qrs_enhanced_v2 | flatline_ratio | 0.5464 | 0.7051 | True | True |
| qrs_enhanced_v2 | amplitude_entropy | 0.4596 | 0.7126 | True | False |
| qrs_enhanced_v2 | template_corr | 0.4588 | 0.6517 | True | False |
| qrs_enhanced_v2 | qrs_band_ratio | 0.4359 | 0.8007 | True | False |
| qrs_enhanced_v2 | pc3 | 0.4230 | 0.7333 | False | True |
| qrs_enhanced_v2 | knn_label_purity | 0.4169 | 0.6895 | False | True |
| qrs_enhanced_v2 | low_amp_ratio | 0.3868 | 0.6589 | False | True |
| qrs_enhanced_v2 | region_confidence | 0.3543 | 0.8064 | True | False |
| qrs_enhanced_v2 | boundary_confidence | 0.3497 | 0.6515 | False | True |
| qrs_enhanced_v2 | baseline_step | 0.3395 | 0.7820 | True | True |
| qrs_enhanced_v2 | detector_agreement | 0.2416 | 0.7533 | True | True |
| qrs_enhanced_v2 | sqi_basSQI | 0.2278 | 0.8696 | True | True |
| qrs_enhanced_v2 | qrs_visibility | 0.1800 | 0.8393 | True | True |
| qrs_enhanced_v2 | pc2 | 0.1450 | 0.9573 | False | True |
| qrs_enhanced_v2 | mean_abs | 0.0583 | 0.8937 | True | False |
| qrs_stress_v3 | pca_margin | 0.7486 | 0.3387 | True | False |
| qrs_stress_v3 | pc1 | 0.7375 | 0.3940 | True | False |
| qrs_stress_v3 | non_qrs_diff_p95 | 0.7243 | 0.4462 | True | False |
| qrs_stress_v3 | sqi_bSQI | 0.6304 | 0.4077 | True | False |
| qrs_stress_v3 | flatline_ratio | 0.5445 | 0.7082 | True | True |
| qrs_stress_v3 | template_corr | 0.4601 | 0.6509 | True | False |
| qrs_stress_v3 | amplitude_entropy | 0.4582 | 0.7110 | True | False |
| qrs_stress_v3 | qrs_band_ratio | 0.4330 | 0.8022 | True | False |
| qrs_stress_v3 | pc3 | 0.4220 | 0.7344 | False | True |
| qrs_stress_v3 | knn_label_purity | 0.4123 | 0.6899 | False | True |
| qrs_stress_v3 | low_amp_ratio | 0.3817 | 0.6625 | False | True |
| qrs_stress_v3 | boundary_confidence | 0.3438 | 0.6525 | False | True |
| qrs_stress_v3 | region_confidence | 0.3435 | 0.8087 | True | False |
| qrs_stress_v3 | baseline_step | 0.3364 | 0.7853 | True | True |
| qrs_stress_v3 | detector_agreement | 0.2644 | 0.7464 | True | True |
| qrs_stress_v3 | sqi_basSQI | 0.2202 | 0.8720 | True | True |
| qrs_stress_v3 | qrs_visibility | 0.1731 | 0.8425 | True | True |
| qrs_stress_v3 | pc2 | 0.1358 | 0.9603 | False | True |
| qrs_stress_v3 | mean_abs | 0.0574 | 0.8956 | True | False |
