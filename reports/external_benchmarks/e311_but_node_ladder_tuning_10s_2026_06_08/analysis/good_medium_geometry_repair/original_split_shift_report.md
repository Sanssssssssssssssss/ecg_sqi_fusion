# Original BUT Split Shift Diagnostic

Report-only diagnostic. No model training or selection uses these results.

## Counts By Split/Class/Region/Record

| split | class_name | original_region | record_id | n |
| --- | --- | --- | --- | --- |
| test | bad | outlier_low_confidence | 111001 | 292 |
| test | bad | near_bad_boundary | 122001 | 119 |
| test | good | outlier_low_confidence | 111001 | 2191 |
| test | good | good_medium_overlap | 111001 | 1128 |
| test | good | outlier_low_confidence | 125001 | 220 |
| test | good | good_medium_overlap | 122001 | 55 |
| test | good | clean_core | 122001 | 21 |
| test | good | clean_core | 111001 | 17 |
| test | good | good_medium_overlap | 125001 | 8 |
| test | medium | outlier_low_confidence | 111001 | 2173 |
| test | medium | good_medium_overlap | 111001 | 1689 |
| test | medium | clean_core | 111001 | 523 |
| test | medium | good_medium_overlap | 122001 | 22 |
| test | medium | clean_core | 122001 | 8 |
| test | medium | outlier_low_confidence | 122001 | 6 |
| test | medium | medium_bad_overlap | 111001 | 5 |
| train | bad | right_bad_island | 105001 | 3964 |
| train | bad | outlier_low_confidence | 105001 | 770 |
| train | bad | outlier_low_confidence | 124001 | 31 |
| train | bad | outlier_low_confidence | 100001 | 22 |
| train | bad | outlier_low_confidence | 113001 | 4 |
| train | good | good_medium_overlap | 100001 | 3692 |
| train | good | good_medium_overlap | 105001 | 3116 |
| train | good | clean_core | 105001 | 1961 |
| train | good | clean_core | 100001 | 1064 |
| train | good | outlier_low_confidence | 100001 | 1030 |
| train | good | outlier_low_confidence | 105001 | 424 |
| train | good | outlier_low_confidence | 104001 | 136 |
| train | good | good_medium_overlap | 113001 | 135 |
| train | good | good_medium_overlap | 118001 | 127 |
| train | good | good_medium_overlap | 121001 | 118 |
| train | good | good_medium_overlap | 115001 | 106 |
| train | good | good_medium_overlap | 123001 | 100 |
| train | good | outlier_low_confidence | 123001 | 85 |
| train | good | outlier_low_confidence | 124001 | 67 |
| train | good | good_medium_overlap | 100002 | 58 |
| train | good | outlier_low_confidence | 113001 | 55 |
| train | good | outlier_low_confidence | 121001 | 36 |
| train | good | clean_core | 100002 | 32 |
| train | good | outlier_low_confidence | 100002 | 24 |

## Top Train-vs-Test Feature Gaps

### good

| class_name | feature | ks_train_vs_test | train_median | test_median | delta_median | train_p10 | train_p90 | test_p10 | test_p90 | n_train | n_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| good | subject_id | 0.927859 | 105 | 111 | 6 | 100 | 105 | 111 | 111 | 12434 | 3640 |
| good | diff_abs_p95 | 0.801525 | 0.192708 | 0.0581305 | -0.134578 | 0.0856152 | 0.346705 | 0.0303409 | 0.0881564 | 12434 | 3640 |
| good | local_rms_cv | 0.769565 | 0.134704 | 0.342812 | 0.208108 | 0.044563 | 0.22318 | 0.193746 | 0.497182 | 12434 | 3640 |
| good | qrs_slope_median | 0.750758 | 3.42904 | 0.208475 | -3.22057 | 0.273207 | 4.04155 | 0.113106 | 0.652998 | 12434 | 3640 |
| good | lf_ratio | 0.741022 | 0.0166109 | 0.231246 | 0.214635 | 0.00329713 | 0.16263 | 0.088918 | 0.530479 | 12434 | 3640 |
| good | band_0p3_1 | 0.741022 | 0.0166109 | 0.231246 | 0.214635 | 0.00329713 | 0.16263 | 0.088918 | 0.530479 | 12434 | 3640 |
| good | qrs_prom_median | 0.732996 | 5.43509 | 1.5991 | -3.83599 | 1.58824 | 6.53276 | 0.926034 | 2.00146 | 12434 | 3640 |
| good | sqi_basSQI | 0.73169 | 0.959274 | 0.780598 | -0.178676 | 0.876766 | 0.985992 | 0.536113 | 0.920882 | 12434 | 3640 |
| good | hjorth_complexity | 0.715429 | 1.69384 | 2.0536 | 0.359764 | 1.58946 | 1.95328 | 1.71289 | 3.68537 | 12434 | 3640 |
| good | pc2 | 0.702995 | -1.48161 | 5.14666 | 6.62827 | -2.92525 | 3.03221 | 0.912408 | 11.6512 | 12434 | 3640 |
| good | qrs_width_median | 0.691356 | 0.0237097 | 0.0917497 | 0.06804 | 0.0211975 | 0.0788169 | 0.0243392 | 0.105246 | 12434 | 3640 |
| good | zero_crossing_rate | 0.678216 | 0.0720576 | 0.0320256 | -0.040032 | 0.0424339 | 0.118495 | 0.0176141 | 0.0576461 | 12434 | 3640 |

### medium

| class_name | feature | ks_train_vs_test | train_median | test_median | delta_median | train_p10 | train_p90 | test_p10 | test_p90 | n_train | n_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medium | subject_id | 0.93833 | 105 | 111 | 6 | 100 | 105 | 111 | 111 | 6097 | 4426 |
| medium | qrs_slope_median | 0.598302 | 3.02518 | 0.510442 | -2.51474 | 1.23714 | 3.63066 | 0.103321 | 3.41974 | 6097 | 4426 |
| medium | qrs_prom_median | 0.594302 | 4.05833 | 1.22731 | -2.83102 | 1.67294 | 5.40066 | 0.60814 | 5.10899 | 6097 | 4426 |
| medium | medium_detail_unreliable_score | 0.582379 | 0.328511 | 0.0542456 | -0.274266 | 0.0756008 | 0.494887 | 0.00533003 | 0.383566 | 6097 | 4426 |
| medium | qrs_visibility | 0.581581 | 0.33452 | 0.0652774 | -0.269243 | 0.0925154 | 0.499138 | 0.0105071 | 0.383647 | 6097 | 4426 |
| medium | sqi_basSQI | 0.512877 | 0.967502 | 0.78413 | -0.183372 | 0.816019 | 0.991406 | 0.292998 | 0.96821 | 6097 | 4426 |
| medium | lf_ratio | 0.51187 | 0.033816 | 0.26544 | 0.231624 | 0.00524507 | 0.222132 | 0.0330099 | 0.767298 | 6097 | 4426 |
| medium | band_0p3_1 | 0.51187 | 0.033816 | 0.26544 | 0.231624 | 0.00524507 | 0.222132 | 0.0330099 | 0.767298 | 6097 | 4426 |
| medium | qrs_band_ratio | 0.506507 | 0.542128 | 0.350001 | -0.192127 | 0.337945 | 0.634453 | 0.0526698 | 0.551973 | 6097 | 4426 |
| medium | band_5_15 | 0.491286 | 0.447242 | 0.295646 | -0.151597 | 0.283734 | 0.533431 | 0.0473618 | 0.462784 | 6097 | 4426 |
| medium | diff_abs_p95 | 0.471871 | 0.288451 | 0.111666 | -0.176785 | 0.143812 | 0.405742 | 0.022 | 0.424706 | 6097 | 4426 |
| medium | template_corr | 0.445521 | 0.579488 | 0.475716 | -0.103772 | 0.477252 | 0.882402 | 0.338893 | 0.932377 | 6097 | 4426 |

### bad

| class_name | feature | ks_train_vs_test | train_median | test_median | delta_median | train_p10 | train_p90 | test_p10 | test_p90 | n_train | n_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad | subject_id | 0.992695 | 105 | 111 | 6 | 105 | 105 | 111 | 122 | 4791 | 411 |
| bad | qrs_band_ratio | 0.988729 | 0.810734 | 0.202367 | -0.608367 | 0.792786 | 0.82627 | 0.0431659 | 0.36101 | 4791 | 411 |
| bad | detail_instability | 0.987059 | 0.22633 | 1.48235 | 1.25602 | 0.219042 | 0.234394 | 0.279343 | 3.5311 | 4791 | 411 |
| bad | sqi_pSQI | 0.987059 | 0.0492123 | 0.686402 | 0.63719 | 0.0418081 | 0.0577933 | 0.306197 | 0.827353 | 4791 | 411 |
| bad | band_1_5 | 0.987059 | 0.0104818 | 0.281669 | 0.271187 | 0.00831029 | 0.0132419 | 0.102039 | 0.511623 | 4791 | 411 |
| bad | band_15_30 | 0.987059 | 0.833946 | 0.05815 | -0.775796 | 0.818782 | 0.848097 | 0.00844915 | 0.335003 | 4791 | 411 |
| bad | wavelet_e2 | 0.987059 | 0.408105 | 0.0441815 | -0.363923 | 0.381504 | 0.433346 | 0.00529067 | 0.212846 | 4791 | 411 |
| bad | pca_own_distance | 0.987059 | 0.645431 | 17.156 | 16.5106 | 0.387072 | 1.10484 | 4.96391 | 25.709 | 4791 | 411 |
| bad | pca_margin | 0.987059 | 10.8818 | -5.64057 | -16.5224 | 10.3466 | 11.203 | -7.59926 | 5.49935 | 4791 | 411 |
| bad | class_margin_percentile | 0.987059 | 0.546831 | 0.048439 | -0.498392 | 0.183917 | 0.909366 | 0.00851466 | 0.0836329 | 4791 | 411 |
| bad | class_centrality_percentile | 0.987059 | 0.546831 | 0.0399243 | -0.506906 | 0.184295 | 0.909366 | 0.00851466 | 0.0836329 | 4791 | 411 |
| bad | pca_margin_rank | 0.987059 | 0.546831 | 0.048439 | -0.498392 | 0.183917 | 0.909366 | 0.00851466 | 0.0836329 | 4791 | 411 |

Counts CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\original_split_shift_counts.csv`
Feature KS CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\original_split_shift_feature_ks.csv`
