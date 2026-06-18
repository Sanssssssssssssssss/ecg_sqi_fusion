# BUT Capacity Error Gap: aug_convtx_balanced_focal_trainval_badcal_t005_visual

Report-only diagnostic. This is not a formal model-selection result.

## Largest Test Buckets

| capacity_bucket | n | acc | good_n | good_recall | medium_n | medium_recall | bad_n | bad_recall | record_id | original_region | true_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_test | 8477 | 0.808659 | 3640 | 0.963187 | 4426 | 0.688884 | 411 | 0.729927 | nan | nan | nan |
| but_test | 8018 | 0.811674 | 3336 | 0.992806 | 4390 | 0.686788 | 292 | 0.619863 | 111001 | nan | nan |
| but_test | 4656 | 0.715421 | 2191 | 0.990872 | 2173 | 0.450529 | 292 | 0.619863 | 111001 | outlier_low_confidence | nan |
| but_test | 2817 | 0.954917 | 1128 | 0.996454 | 1689 | 0.927176 | 0 | nan | 111001 | good_medium_overlap | nan |
| but_test | 2191 | 0.990872 | 2191 | 0.990872 | 0 | nan | 0 | nan | 111001 | outlier_low_confidence | good |
| but_test | 2173 | 0.450529 | 0 | nan | 2173 | 0.450529 | 0 | nan | 111001 | outlier_low_confidence | medium |
| but_test | 1689 | 0.927176 | 0 | nan | 1689 | 0.927176 | 0 | nan | 111001 | good_medium_overlap | medium |
| but_test | 1128 | 0.996454 | 1128 | 0.996454 | 0 | nan | 0 | nan | 111001 | good_medium_overlap | good |
| but_test | 540 | 0.9 | 17 | 1 | 523 | 0.89675 | 0 | nan | 111001 | clean_core | nan |
| but_test | 523 | 0.89675 | 0 | nan | 523 | 0.89675 | 0 | nan | 111001 | clean_core | medium |
| but_test | 292 | 0.619863 | 0 | nan | 0 | nan | 292 | 0.619863 | 111001 | outlier_low_confidence | bad |
| but_test | 231 | 0.991342 | 76 | 1 | 36 | 0.944444 | 119 | 1 | 122001 | nan | nan |
| but_test | 228 | 0.517544 | 228 | 0.517544 | 0 | nan | 0 | nan | 125001 | nan | nan |
| but_test | 220 | 0.504545 | 220 | 0.504545 | 0 | nan | 0 | nan | 125001 | outlier_low_confidence | nan |
| but_test | 220 | 0.504545 | 220 | 0.504545 | 0 | nan | 0 | nan | 125001 | outlier_low_confidence | good |
| but_test | 119 | 1 | 0 | nan | 0 | nan | 119 | 1 | 122001 | near_bad_boundary | nan |
| but_test | 119 | 1 | 0 | nan | 0 | nan | 119 | 1 | 122001 | near_bad_boundary | bad |
| but_test | 77 | 1 | 55 | 1 | 22 | 1 | 0 | nan | 122001 | good_medium_overlap | nan |
| but_test | 55 | 1 | 55 | 1 | 0 | nan | 0 | nan | 122001 | good_medium_overlap | good |
| but_test | 29 | 1 | 21 | 1 | 8 | 1 | 0 | nan | 122001 | clean_core | nan |
| but_test | 22 | 1 | 0 | nan | 22 | 1 | 0 | nan | 122001 | good_medium_overlap | medium |
| but_test | 21 | 1 | 21 | 1 | 0 | nan | 0 | nan | 122001 | clean_core | good |
| but_test | 17 | 1 | 17 | 1 | 0 | nan | 0 | nan | 111001 | clean_core | good |
| but_test | 8 | 0.875 | 8 | 0.875 | 0 | nan | 0 | nan | 125001 | good_medium_overlap | good |
| but_test | 8 | 0.875 | 8 | 0.875 | 0 | nan | 0 | nan | 125001 | good_medium_overlap | nan |

## Top Test Error Transitions

| capacity_bucket | record_id | original_region | true_label | pred_label | n |
| --- | --- | --- | --- | --- | --- |
| but_test | 111001 | outlier_low_confidence | good | good | 2171 |
| but_test | 111001 | good_medium_overlap | medium | medium | 1566 |
| but_test | 111001 | good_medium_overlap | good | good | 1124 |
| but_test | 111001 | outlier_low_confidence | medium | good | 1023 |
| but_test | 111001 | outlier_low_confidence | medium | medium | 979 |
| but_test | 111001 | clean_core | medium | medium | 469 |
| but_test | 111001 | outlier_low_confidence | bad | bad | 181 |
| but_test | 111001 | outlier_low_confidence | medium | bad | 171 |
| but_test | 122001 | near_bad_boundary | bad | bad | 119 |
| but_test | 125001 | outlier_low_confidence | good | good | 111 |
| but_test | 125001 | outlier_low_confidence | good | medium | 104 |
| but_test | 111001 | outlier_low_confidence | bad | medium | 91 |
| but_test | 111001 | good_medium_overlap | medium | good | 77 |
| but_test | 122001 | good_medium_overlap | good | good | 55 |
| but_test | 111001 | clean_core | medium | bad | 54 |
| but_test | 111001 | good_medium_overlap | medium | bad | 46 |
| but_test | 122001 | good_medium_overlap | medium | medium | 22 |
| but_test | 122001 | clean_core | good | good | 21 |
| but_test | 111001 | outlier_low_confidence | bad | good | 20 |
| but_test | 111001 | outlier_low_confidence | good | medium | 20 |
| but_test | 111001 | clean_core | good | good | 17 |
| but_test | 122001 | clean_core | medium | medium | 8 |
| but_test | 125001 | good_medium_overlap | good | good | 7 |
| but_test | 125001 | outlier_low_confidence | good | bad | 5 |
| but_test | 111001 | medium_bad_overlap | medium | bad | 4 |

## Top Correct-vs-Wrong Feature Gaps

| class_name | feature | ks_correct_vs_wrong | correct_median | wrong_median | delta_wrong_minus_correct | correct_p10 | correct_p90 | wrong_p10 | wrong_p90 | n_correct | n_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad | rms | 0.510811 | 0.163107 | 0.181757 | 0.0186504 | 0.151866 | 0.197654 | 0.163542 | 0.213143 | 300 | 111 |
| bad | std | 0.467568 | 0.162754 | 0.176183 | 0.0134288 | 0.150167 | 0.195557 | 0.163432 | 0.211417 | 300 | 111 |
| bad | hjorth_activity | 0.467568 | 0.0264889 | 0.0310404 | 0.00455151 | 0.0225502 | 0.0382425 | 0.0267101 | 0.044697 | 300 | 111 |
| bad | pc7 | 0.46027 | 2.13218 | -0.449895 | -2.58207 | -1.50404 | 3.62273 | -2.85339 | 1.16519 | 300 | 111 |
| bad | boundary_confidence | 0.44 | 0.0294087 | 0.022649 | -0.0067597 | 0.0135809 | 0.392886 | 0.00943236 | 0.0311164 | 300 | 111 |
| bad | region_confidence | 0.44 | 0.00735218 | 0.00566225 | -0.00168992 | 0.00339522 | 0.306451 | 0.00235809 | 0.00777909 | 300 | 111 |
| bad | knn_label_purity | 0.430991 | 0 | 0 | 0 | 0 | 0.966667 | 0 | 0 | 300 | 111 |
| bad | non_qrs_diff_p95 | 0.43 | 0.0920653 | 0.0347303 | -0.057335 | 0.0210196 | 0.431502 | 0.0187098 | 0.099614 | 300 | 111 |
| bad | local_rms_cv | 0.426306 | 0.392488 | 0.527775 | 0.135287 | 0.051517 | 0.867393 | 0.363681 | 0.711746 | 300 | 111 |
| bad | detector_count_std | 0.426126 | 1.69967 | 2.86744 | 1.16777 | 0.471405 | 3.26599 | 1.63299 | 4.10961 | 300 | 111 |
| bad | detector_agreement | 0.426126 | 0.370415 | 0.258569 | -0.111846 | 0.234412 | 0.679623 | 0.19571 | 0.379796 | 300 | 111 |
| bad | pca_own_distance | 0.424955 | 15.4036 | 18.6555 | 3.25198 | 4.90297 | 24.7886 | 14.626 | 27.4804 | 300 | 111 |
| bad | class_centrality_percentile | 0.424955 | 0.0469253 | 0.0321665 | -0.0147588 | 0.0101798 | 0.0857332 | 0.00662252 | 0.0505203 | 300 | 111 |
| bad | own_centrality_rank | 0.424955 | 0.0469253 | 0.0321665 | -0.0147588 | 0.0101798 | 0.0857332 | 0.00662252 | 0.0505203 | 300 | 111 |
| bad | lf_ratio | 0.423333 | 0.237624 | 0.438304 | 0.20068 | 0.0127702 | 0.562433 | 0.215037 | 0.795103 | 300 | 111 |
| bad | rr_count_detector_c | 0.423333 | 18 | 13 | -5 | 10 | 28 | 9 | 18 | 300 | 111 |
| bad | band_0p3_1 | 0.423333 | 0.237624 | 0.438304 | 0.20068 | 0.0127702 | 0.562433 | 0.215037 | 0.795103 | 300 | 111 |
| bad | pc1 | 0.416667 | -1.80964 | -4.00739 | -2.19775 | -6.51928 | 9.21193 | -5.61697 | -1.27334 | 300 | 111 |
| bad | baseline_step | 0.413333 | 0.968748 | 1.48359 | 0.514846 | 0.194219 | 1.82147 | 0.849887 | 2.25427 | 300 | 111 |
| bad | sqi_basSQI | 0.413333 | 0.771821 | 0.60401 | -0.167811 | 0.476397 | 0.985182 | 0.266473 | 0.814764 | 300 | 111 |
| bad | zero_crossing_rate | 0.413333 | 0.0592474 | 0.0288231 | -0.0304243 | 0.0208167 | 0.502962 | 0.0136109 | 0.0672538 | 300 | 111 |
| bad | diff_zero_crossing_rate | 0.408649 | 0.403846 | 0.317308 | -0.0865385 | 0.221154 | 0.681971 | 0.229968 | 0.419872 | 300 | 111 |
| bad | diff_abs_median | 0.406667 | 0.0066581 | 0.00395533 | -0.00270277 | 0.0017194 | 0.158394 | 0.0022891 | 0.00752016 | 300 | 111 |
| bad | sample_entropy_proxy | 0.406667 | 0.519301 | 0.389389 | -0.129912 | 0.259544 | 0.900435 | 0.204964 | 0.550956 | 300 | 111 |
| bad | wavelet_e2 | 0.406667 | 0.0577225 | 0.026986 | -0.0307366 | 0.00949792 | 0.218702 | 0.00189953 | 0.0888071 | 300 | 111 |
| bad | flatline_ratio | 0.403333 | 0.269416 | 0.393915 | 0.1245 | 0.00720576 | 0.623058 | 0.240192 | 0.560448 | 300 | 111 |
| bad | diff_abs_p95 | 0.403333 | 0.131584 | 0.0785064 | -0.0530779 | 0.0429435 | 0.445043 | 0.0254453 | 0.154069 | 300 | 111 |
| bad | qrs_slope_median | 0.403333 | 0.747593 | 0.423309 | -0.324284 | 0.164603 | 3.0041 | 0.120599 | 1.08098 | 300 | 111 |
| bad | rr_count_detector_b | 0.403333 | 19 | 16 | -3 | 12.9 | 28 | 12 | 20 | 300 | 111 |
| bad | wavelet_e0 | 0.403333 | 0.772247 | 0.881224 | 0.108977 | 0.211474 | 0.947009 | 0.706324 | 0.988523 | 300 | 111 |
| bad | spurious_peak_density | 0.402252 | 3.85 | 2.4 | -1.45 | 1.3 | 6.6 | 1.4 | 4 | 300 | 111 |
| bad | qrs_prom_median | 0.401622 | 1.50699 | 0.98418 | -0.522815 | 0.564875 | 2.41813 | 0.474156 | 1.72286 | 300 | 111 |
| bad | detail_instability | 0.397658 | 1.34198 | 1.82812 | 0.48614 | 0.27652 | 3.62513 | 0.881058 | 3.4007 | 300 | 111 |
| bad | spectral_entropy | 0.397658 | 0.72163 | 0.632529 | -0.0891009 | 0.528286 | 0.930745 | 0.38202 | 0.77807 | 300 | 111 |
| bad | band_30_45 | 0.397658 | 0.0219506 | 0.011203 | -0.0107476 | 0.00246427 | 0.296703 | 0.00155145 | 0.0397015 | 300 | 111 |

Summary CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\aug_convtx_balanced_focal_trainval_badcal_t005_visual_summary_by_bucket_record_region.csv`
Transitions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\aug_convtx_balanced_focal_trainval_badcal_t005_visual_transitions.csv`
Feature gaps CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\aug_convtx_balanced_focal_trainval_badcal_t005_visual_test_error_feature_gaps.csv`
Test errors CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\aug_convtx_balanced_focal_trainval_badcal_t005_visual_test_errors.csv`
Waveform panel: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\aug_convtx_balanced_focal_trainval_badcal_t005_visual_test_error_waveform_panels.png`
