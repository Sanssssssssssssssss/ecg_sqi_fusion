# Waveform Candidate Error Exchange

Scope: BUT original test split only. This is report-only analysis; no BUT rows are used for training or selection.

Reference candidate: `featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050`

## featurefirst_top20_qrsbase_primres_current_conservative_a050

| exchange | n |
|---|---:|
| candidate_fixes_reference | 69 |
| candidate_regresses_reference | 355 |
| both_correct | 6967 |
| both_wrong | 1086 |

Largest class/region exchanges:

| exchange | true | region | n | ref_pred_top | cand_pred_top | record_top |
| --- | --- | --- | --- | --- | --- | --- |
| both_correct | good | outlier_low_confidence | 2058 | good | good | 111001 |
| both_correct | medium | good_medium_overlap | 1694 | medium | medium | 111001 |
| both_correct | medium | outlier_low_confidence | 1261 | medium | medium | 111001 |
| both_correct | good | good_medium_overlap | 1142 | good | good | 111001 |
| both_correct | medium | clean_core | 530 | medium | medium | 111001 |
| both_correct | bad | outlier_low_confidence | 124 | bad | bad | 111001 |
| both_correct | bad | near_bad_boundary | 115 | bad | bad | 122001 |
| both_correct | good | clean_core | 38 | good | good | 122001 |
| both_correct | medium | medium_bad_overlap | 5 | medium | medium | 111001 |
| both_wrong | medium | outlier_low_confidence | 588 | good | good | 111001 |
| both_wrong | good | outlier_low_confidence | 300 | medium | medium | 125001 |
| both_wrong | bad | outlier_low_confidence | 146 | good | good | 111001 |

Top feature differences: fixed p20 errors vs regressed p20 correct rows

| feature | fixes_median | regresses_median | median_delta_a_minus_b | abs_standardized_delta | fixes_n | regresses_n |
| --- | --- | --- | --- | --- | --- | --- |
| knn_label_purity | 0.933333 | 0.0666667 | 0.866667 | 2.52297 | 69 | 355 |
| boundary_confidence | 0.426712 | 0.0840315 | 0.34268 | 2.14686 | 69 | 355 |
| region_confidence | 0.108401 | 0.0210079 | 0.0873929 | 1.11144 | 69 | 355 |
| baseline_step | 1.18657 | 1.72123 | -0.534656 | 0.906631 | 69 | 355 |
| sqi_bSQI | 0.857143 | 0.466667 | 0.390476 | 0.890052 | 69 | 355 |
| hjorth_complexity | 4.4245 | 8.0253 | -3.6008 | 0.863855 | 69 | 355 |
| sqi_basSQI | 0.533199 | 0.343966 | 0.189233 | 0.830096 | 69 | 355 |
| band_5_15 | 0.168536 | 0.0685768 | 0.0999596 | 0.745851 | 69 | 355 |
| pc2 | 12.5932 | 16.4659 | -3.87277 | 0.731658 | 69 | 355 |
| spectral_entropy | 0.579824 | 0.470201 | 0.109623 | 0.720892 | 69 | 355 |
| qrs_band_ratio | 0.18296 | 0.0769378 | 0.106023 | 0.719646 | 69 | 355 |
| lf_ratio | 0.541019 | 0.705862 | -0.164843 | 0.672357 | 69 | 355 |
| band_0p3_1 | 0.541019 | 0.705862 | -0.164843 | 0.672357 | 69 | 355 |
| template_corr | 0.499198 | 0.4218 | 0.077398 | 0.664741 | 69 | 355 |
| qrs_visibility | 0.055897 | 0.0205313 | 0.0353658 | 0.659234 | 69 | 355 |

![featurefirst_top20_qrsbase_primres_current_conservative_a050 exchange waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_candidate_exchange\featurefirst_top20_qrsbase_primres_current_conservative_a050_exchange_waveforms.png)


## featurefirst_top20_qrsbase_primres_current_balanced_a050

| exchange | n |
|---|---:|
| candidate_fixes_reference | 74 |
| candidate_regresses_reference | 396 |
| both_correct | 6926 |
| both_wrong | 1081 |

Largest class/region exchanges:

| exchange | true | region | n | ref_pred_top | cand_pred_top | record_top |
| --- | --- | --- | --- | --- | --- | --- |
| both_correct | good | outlier_low_confidence | 2058 | good | good | 111001 |
| both_correct | medium | good_medium_overlap | 1695 | medium | medium | 111001 |
| both_correct | medium | outlier_low_confidence | 1259 | medium | medium | 111001 |
| both_correct | good | good_medium_overlap | 1142 | good | good | 111001 |
| both_correct | medium | clean_core | 530 | medium | medium | 111001 |
| both_correct | bad | near_bad_boundary | 114 | bad | bad | 122001 |
| both_correct | bad | outlier_low_confidence | 85 | bad | bad | 111001 |
| both_correct | good | clean_core | 38 | good | good | 122001 |
| both_correct | medium | medium_bad_overlap | 5 | medium | medium | 111001 |
| both_wrong | medium | outlier_low_confidence | 586 | good | good | 111001 |
| both_wrong | good | outlier_low_confidence | 293 | medium | medium | 125001 |
| both_wrong | bad | outlier_low_confidence | 148 | good | good | 111001 |

Top feature differences: fixed p20 errors vs regressed p20 correct rows

| feature | fixes_median | regresses_median | median_delta_a_minus_b | abs_standardized_delta | fixes_n | regresses_n |
| --- | --- | --- | --- | --- | --- | --- |
| knn_label_purity | 0.933333 | 0.05 | 0.883333 | 2.87436 | 74 | 396 |
| boundary_confidence | 0.431928 | 0.0781404 | 0.353787 | 2.38443 | 74 | 396 |
| region_confidence | 0.108218 | 0.0195351 | 0.0886829 | 0.994872 | 74 | 396 |
| sqi_bSQI | 0.857143 | 0.464103 | 0.39304 | 0.985432 | 74 | 396 |
| template_corr | 0.502634 | 0.409136 | 0.0934986 | 0.889321 | 74 | 396 |
| baseline_step | 1.25829 | 1.70669 | -0.4484 | 0.789022 | 74 | 396 |
| hjorth_complexity | 4.48397 | 7.48466 | -3.0007 | 0.715374 | 74 | 396 |
| sample_entropy_proxy | 0.349707 | 0.416978 | -0.0672707 | 0.695236 | 74 | 396 |
| band_5_15 | 0.162378 | 0.073918 | 0.0884603 | 0.648198 | 74 | 396 |
| sqi_basSQI | 0.531225 | 0.386557 | 0.144668 | 0.640491 | 74 | 396 |
| qrs_band_ratio | 0.177168 | 0.0806798 | 0.096488 | 0.624665 | 74 | 396 |
| pc2 | 12.677 | 15.8509 | -3.17396 | 0.583582 | 74 | 396 |
| mean_abs | 0.133986 | 0.123269 | 0.0107166 | 0.574441 | 74 | 396 |
| spectral_entropy | 0.58135 | 0.48514 | 0.0962101 | 0.564484 | 74 | 396 |
| diff_zero_crossing_rate | 0.340144 | 0.298077 | 0.0420673 | 0.531585 | 74 | 396 |

![featurefirst_top20_qrsbase_primres_current_balanced_a050 exchange waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_candidate_exchange\featurefirst_top20_qrsbase_primres_current_balanced_a050_exchange_waveforms.png)


## featurefirst_top20_qrsbase_primres_current_badlean_a050

| exchange | n |
|---|---:|
| candidate_fixes_reference | 64 |
| candidate_regresses_reference | 345 |
| both_correct | 6977 |
| both_wrong | 1091 |

Largest class/region exchanges:

| exchange | true | region | n | ref_pred_top | cand_pred_top | record_top |
| --- | --- | --- | --- | --- | --- | --- |
| both_correct | good | outlier_low_confidence | 2058 | good | good | 111001 |
| both_correct | medium | good_medium_overlap | 1697 | medium | medium | 111001 |
| both_correct | medium | outlier_low_confidence | 1310 | medium | medium | 111001 |
| both_correct | good | good_medium_overlap | 1142 | good | good | 111001 |
| both_correct | medium | clean_core | 530 | medium | medium | 111001 |
| both_correct | bad | near_bad_boundary | 112 | bad | bad | 122001 |
| both_correct | bad | outlier_low_confidence | 85 | bad | bad | 111001 |
| both_correct | good | clean_core | 38 | good | good | 122001 |
| both_correct | medium | medium_bad_overlap | 5 | medium | medium | 111001 |
| both_wrong | medium | outlier_low_confidence | 581 | good | good | 111001 |
| both_wrong | good | outlier_low_confidence | 303 | medium | medium | 125001 |
| both_wrong | bad | outlier_low_confidence | 149 | good | good | 111001 |

Top feature differences: fixed p20 errors vs regressed p20 correct rows

| feature | fixes_median | regresses_median | median_delta_a_minus_b | abs_standardized_delta | fixes_n | regresses_n |
| --- | --- | --- | --- | --- | --- | --- |
| knn_label_purity | 0.933333 | 0.0333333 | 0.9 | 2.61681 | 64 | 345 |
| boundary_confidence | 0.430879 | 0.0688687 | 0.36201 | 2.31828 | 64 | 345 |
| region_confidence | 0.107982 | 0.0172172 | 0.0907648 | 0.927535 | 64 | 345 |
| sqi_bSQI | 0.845238 | 0.454545 | 0.390693 | 0.839614 | 64 | 345 |
| template_corr | 0.495137 | 0.399815 | 0.0953223 | 0.758865 | 64 | 345 |
| pca_margin | 0.26201 | -0.122782 | 0.384792 | 0.653843 | 64 | 345 |
| baseline_step | 1.30373 | 1.71855 | -0.414817 | 0.606518 | 64 | 345 |
| sample_entropy_proxy | 0.359052 | 0.417866 | -0.0588144 | 0.544926 | 64 | 345 |
| hjorth_complexity | 5.14821 | 7.60328 | -2.45507 | 0.534307 | 64 | 345 |
| mean_abs | 0.13309 | 0.123044 | 0.010046 | 0.497298 | 64 | 345 |
| detail_instability | 0.655787 | 0.648073 | 0.00771416 | 0.489654 | 64 | 345 |
| band_5_15 | 0.135151 | 0.070087 | 0.0650643 | 0.489138 | 64 | 345 |
| diff_zero_crossing_rate | 0.325721 | 0.290064 | 0.0356571 | 0.48152 | 64 | 345 |
| qrs_band_ratio | 0.151309 | 0.0773885 | 0.0739203 | 0.467356 | 64 | 345 |
| sqi_basSQI | 0.48405 | 0.374101 | 0.109949 | 0.460194 | 64 | 345 |

![featurefirst_top20_qrsbase_primres_current_badlean_a050 exchange waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_candidate_exchange\featurefirst_top20_qrsbase_primres_current_badlean_a050_exchange_waveforms.png)

