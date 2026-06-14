# Current Reduced SQI/Geometry Feature Candidates

Reduced-feature tabular MLP candidates trained with train split only, selected on validation only, and reported on the held-out BUT test split.

## Feature Ranking Top 16

| rank | feature | ranking_score | perm_acc_drop_val_mean | train_gm_auc | val_gm_auc | train_bad_auc | val_bad_auc | train_gm_direction | train_bad_direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | pca_margin | 2.871992 | 0.155575 | 0.522710 | 0.650135 | 0.987405 | 0.921496 | higher_positive | higher_positive |
| 2 | sample_entropy_proxy | 2.551441 | 0.020570 | 0.945487 | 0.954838 | 0.999494 | 0.994918 | higher_positive | higher_positive |
| 3 | flatline_ratio | 2.455340 | 0.014693 | 0.904003 | 0.923677 | 0.998584 | 0.993578 | lower_positive | lower_positive |
| 4 | pc1 | 2.362666 | 0.002420 | 0.936987 | 0.871630 | 0.998246 | 0.996365 | higher_positive | higher_positive |
| 5 | higuchi_fd_proxy | 2.360232 | 0.011409 | 0.883455 | 0.823205 | 0.991528 | 0.988737 | higher_positive | higher_positive |
| 6 | non_qrs_diff_p95 | 2.244800 | 0.010545 | 0.831775 | 0.659040 | 0.997932 | 0.999731 | higher_positive | higher_positive |
| 7 | diff_zero_crossing_rate | 2.209242 | 0.000691 | 0.756717 | 0.922006 | 0.986846 | 0.998861 | higher_positive | higher_positive |
| 8 | zero_crossing_rate | 2.172992 | 0.007087 | 0.785063 | 0.681675 | 0.995117 | 0.998676 | higher_positive | higher_positive |
| 9 | qrs_prom_p90 | 2.156201 | 0.002939 | 0.783426 | 0.806546 | 0.965237 | 0.994369 | lower_positive | lower_positive |
| 10 | sqi_fSQI | 2.117275 | 0.001037 | 0.794097 | 0.779974 | 0.967311 | 0.977760 | lower_positive | lower_positive |
| 11 | amplitude_entropy | 2.110859 | 0.001383 | 0.755257 | 0.800885 | 0.988674 | 0.947814 | higher_positive | higher_positive |
| 12 | sqi_kSQI | 2.109859 | 0.001037 | 0.777323 | 0.751044 | 0.977054 | 0.958650 | lower_positive | lower_positive |
| 13 | sqi_sSQI | 2.100312 | 0.003111 | 0.782195 | 0.687759 | 0.969853 | 0.961881 | lower_positive | lower_positive |
| 14 | low_amp_ratio | 2.090229 | 0.000000 | 0.764073 | 0.818723 | 0.946861 | 0.953002 | lower_positive | lower_positive |
| 15 | non_qrs_rms_ratio | 2.085509 | 0.000519 | 0.746986 | 0.717726 | 0.992256 | 0.983476 | higher_positive | higher_positive |
| 16 | baseline_step | 2.062861 | 0.001037 | 0.701032 | 0.815775 | 0.991744 | 0.919454 | higher_positive | lower_positive |

## Train/Val Selection

| candidate | n_features | features | val_acc | val_macro_f1 | val_good_recall | val_medium_recall | val_bad_recall | val_score | original_test_acc_report_only | original_test_good_recall | original_test_medium_recall | original_test_bad_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| auto_topk | 20 | pca_margin, sample_entropy_proxy, flatline_ratio, pc1, higuchi_fd_proxy, non_qrs_diff_p95, diff_zero_crossing_rate, zero_crossing_rate, qrs_prom_p90, sqi_fSQI, amplitude_entropy, sqi_kSQI, sqi_sSQI, low_amp_ratio, non_qrs_rms_ratio, baseline_step, ptp_p99_p01, band_30_45, qrs_visibility, band_15_30 | 0.996543 | 0.991232 | 0.996904 | 1.000000 | 0.987952 | 1.640024 | 0.913531 | 0.973352 | 0.884546 | 0.695864 |
| ranked_top22 | 22 | pca_margin, sample_entropy_proxy, flatline_ratio, pc1, higuchi_fd_proxy, non_qrs_diff_p95, diff_zero_crossing_rate, zero_crossing_rate, qrs_prom_p90, sqi_fSQI, amplitude_entropy, sqi_kSQI, sqi_sSQI, low_amp_ratio, non_qrs_rms_ratio, baseline_step, ptp_p99_p01, band_30_45, qrs_visibility, band_15_30, hjorth_mobility, wavelet_e4 | 0.996543 | 0.991174 | 0.997936 | 0.990476 | 0.987952 | 1.640000 | 0.908576 | 0.993956 | 0.860822 | 0.666667 |
| ranked_top24 | 24 | pca_margin, sample_entropy_proxy, flatline_ratio, pc1, higuchi_fd_proxy, non_qrs_diff_p95, diff_zero_crossing_rate, zero_crossing_rate, qrs_prom_p90, sqi_fSQI, amplitude_entropy, sqi_kSQI, sqi_sSQI, low_amp_ratio, non_qrs_rms_ratio, baseline_step, ptp_p99_p01, band_30_45, qrs_visibility, band_15_30, hjorth_mobility, wavelet_e4, pc3, rms | 0.996543 | 0.991054 | 1.000000 | 0.971429 | 0.987952 | 1.635822 | 0.890173 | 0.980495 | 0.851333 | 0.508516 |
| top14_balanced | 14 | pc1, qrs_visibility, baseline_step, flatline_ratio, non_qrs_diff_p95, pca_margin, qrs_band_ratio, template_corr, amplitude_entropy, sqi_bSQI, region_confidence, detector_agreement, mean_abs, sqi_basSQI | 0.993086 | 0.984427 | 0.993808 | 0.990476 | 0.987952 | 1.633844 | 0.891707 | 0.892308 | 0.898780 | 0.810219 |
| ranked_top20 | 20 | pca_margin, sample_entropy_proxy, flatline_ratio, pc1, higuchi_fd_proxy, non_qrs_diff_p95, diff_zero_crossing_rate, zero_crossing_rate, qrs_prom_p90, sqi_fSQI, amplitude_entropy, sqi_kSQI, sqi_sSQI, low_amp_ratio, non_qrs_rms_ratio, baseline_step, ptp_p99_p01, band_30_45, qrs_visibility, band_15_30 | 0.987900 | 0.973523 | 0.993808 | 0.933333 | 0.987952 | 1.610642 | 0.839920 | 0.921154 | 0.824446 | 0.287105 |
| top10_interpretable | 10 | pc1, qrs_visibility, baseline_step, flatline_ratio, non_qrs_diff_p95, pca_margin, qrs_band_ratio, template_corr, amplitude_entropy, sqi_bSQI | 0.987035 | 0.971875 | 0.992776 | 0.933333 | 0.987952 | 1.609119 | 0.811490 | 0.842308 | 0.830999 | 0.328467 |
| top6_core | 6 | pc1, qrs_visibility, baseline_step, flatline_ratio, non_qrs_diff_p95, pca_margin | 0.980121 | 0.947817 | 0.993808 | 0.904762 | 0.915663 | 1.585438 | 0.795800 | 0.940110 | 0.725034 | 0.279805 |
| ranked_top18 | 18 | pca_margin, sample_entropy_proxy, flatline_ratio, pc1, higuchi_fd_proxy, non_qrs_diff_p95, diff_zero_crossing_rate, zero_crossing_rate, qrs_prom_p90, sqi_fSQI, amplitude_entropy, sqi_kSQI, sqi_sSQI, low_amp_ratio, non_qrs_rms_ratio, baseline_step, ptp_p99_p01, band_30_45 | 0.977528 | 0.947951 | 0.990712 | 0.876190 | 0.951807 | 1.575756 | 0.796862 | 0.926374 | 0.737460 | 0.289538 |

## Held-Out Original BUT Test

| Candidate | n features | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto_topk | 20 | 0.913531 | 0.885337 | 0.973352 | 0.884546 | 0.695864 | 97 | 511 | 123 |
| ranked_top22 | 22 | 0.908576 | 0.875538 | 0.993956 | 0.860822 | 0.666667 | 22 | 616 | 132 |
| ranked_top24 | 24 | 0.890173 | 0.823727 | 0.980495 | 0.851333 | 0.508516 | 71 | 658 | 159 |
| top14_balanced | 14 | 0.891707 | 0.891574 | 0.892308 | 0.898780 | 0.810219 | 392 | 446 | 61 |
| ranked_top20 | 20 | 0.839920 | 0.716611 | 0.921154 | 0.824446 | 0.287105 | 286 | 777 | 12 |
| top10_interpretable | 10 | 0.811490 | 0.711015 | 0.842308 | 0.830999 | 0.328467 | 574 | 745 | 0 |
| top6_core | 6 | 0.795800 | 0.677465 | 0.940110 | 0.725034 | 0.279805 | 218 | 1188 | 13 |
| ranked_top18 | 18 | 0.796862 | 0.685885 | 0.926374 | 0.737460 | 0.289538 | 268 | 1151 | 26 |
| 47-feature current best, threshold p_bad>=0.13 | 47 | 0.963548 | 0.930683 | 0.956319 | 0.972887 | 0.927007 | 102 | 76 | 23 |
| Current 7SQI-only baseline | 7 | 0.635720 | 0.388833 | 0.275275 | 0.991188 | 0.000000 | 2638 | 39 | 365 |

## Train+Val Bad-Threshold Ablation

This is a separate explanatory ablation, not the pure reduced-feature candidate. The threshold is selected on train+val only and then reported on held-out test.

| Candidate | Threshold | Acc | Macro-F1 | Good R | Medium R | Bad R |
|---|---:|---:|---:|---:|---:|---:|
| auto_topk_badcal | 0.29 | 0.915182 | 0.893222 | 0.973352 | 0.884546 | 0.729927 |
| ranked_top22_badcal | 0.37 | 0.910228 | 0.884056 | 0.993956 | 0.860822 | 0.700730 |
| ranked_top24_badcal | 0.07 | 0.895246 | 0.856491 | 0.976099 | 0.850655 | 0.659367 |
| top14_balanced_badcal | 0.09 | 0.898431 | 0.917485 | 0.892308 | 0.896521 | 0.973236 |
| ranked_top20_badcal | 0.48 | 0.840156 | 0.718629 | 0.921154 | 0.824446 | 0.291971 |
| top10_interpretable_badcal | 0.45 | 0.811962 | 0.714788 | 0.842308 | 0.830999 | 0.338200 |
| top6_core_badcal | 0.49 | 0.795800 | 0.677465 | 0.940110 | 0.725034 | 0.279805 |
| ranked_top18_badcal | 0.50 | 0.796862 | 0.685885 | 0.926374 | 0.737460 | 0.289538 |

## Original Buckets

| candidate | bucket | n | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_precision | medium_precision | bad_precision | good_to_medium | medium_to_good | bad_to_medium | confusion_3x3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top6_core | original_test_all_10s+ | 8477 | 0.795800 | 0.677465 | 0.940110 | 0.725034 | 0.279805 | 0.699366 | 0.932849 | 0.798611 | 218 | 1188 | 13 | [[3422, 218, 0], [1188, 3209, 29], [283, 13, 115]] |
| top6_core | original_all_10s+ | 32956 | 0.878626 | 0.884998 | 0.930059 | 0.770041 | 0.931126 | 0.862030 | 0.867961 | 0.957579 | 1192 | 2226 | 53 | [[15851, 1192, 0], [2226, 8184, 218], [311, 53, 4921]] |
| top6_core | original_test_without_bad_outlier_stress | 8185 | 0.824191 | 0.840317 | 0.940110 | 0.725034 | 0.966387 | 0.742299 | 0.935296 | 0.798611 | 218 | 1188 | 4 | [[3422, 218, 0], [1188, 3209, 29], [0, 4, 115]] |
| top6_core | bad_core_nearboundary | 119 | 0.966387 | 0.327635 | 0.000000 | 0.000000 | 0.966387 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 4 | [[0, 0, 0], [0, 0, 0], [0, 4, 115]] |
| top6_core | bad_outlier_stress | 292 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 9 | [[0, 0, 0], [0, 0, 0], [283, 9, 0]] |
| top6_core_badcal | original_test_all_10s+ | 8477 | 0.795800 | 0.677465 | 0.940110 | 0.725034 | 0.279805 | 0.699366 | 0.932849 | 0.798611 | 218 | 1188 | 13 | [[3422, 218, 0], [1188, 3209, 29], [283, 13, 115]] |
| top6_core_badcal | original_all_10s+ | 32956 | 0.878626 | 0.884998 | 0.930059 | 0.770041 | 0.931126 | 0.862030 | 0.867961 | 0.957579 | 1192 | 2226 | 53 | [[15851, 1192, 0], [2226, 8184, 218], [311, 53, 4921]] |
| top6_core_badcal | original_test_without_bad_outlier_stress | 8185 | 0.824191 | 0.840317 | 0.940110 | 0.725034 | 0.966387 | 0.742299 | 0.935296 | 0.798611 | 218 | 1188 | 4 | [[3422, 218, 0], [1188, 3209, 29], [0, 4, 115]] |
| top6_core_badcal | bad_core_nearboundary | 119 | 0.966387 | 0.327635 | 0.000000 | 0.000000 | 0.966387 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 4 | [[0, 0, 0], [0, 0, 0], [0, 4, 115]] |
| top6_core_badcal | bad_outlier_stress | 292 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 9 | [[0, 0, 0], [0, 0, 0], [283, 9, 0]] |
| top10_interpretable | original_test_all_10s+ | 8477 | 0.811490 | 0.711015 | 0.842308 | 0.830999 | 0.328467 | 0.750184 | 0.865005 | 0.978261 | 574 | 745 | 0 | [[3066, 574, 0], [745, 3678, 3], [276, 0, 135]] |
| top10_interpretable | original_all_10s+ | 32956 | 0.916646 | 0.924350 | 0.946899 | 0.854065 | 0.944939 | 0.898202 | 0.909610 | 0.996806 | 898 | 1542 | 4 | [[16138, 898, 7], [1542, 9077, 9], [287, 4, 4994]] |
| top10_interpretable | original_test_without_bad_outlier_stress | 8185 | 0.838485 | 0.886063 | 0.842308 | 0.830999 | 1.000000 | 0.804513 | 0.865005 | 0.975410 | 574 | 745 | 0 | [[3066, 574, 0], [745, 3678, 3], [0, 0, 119]] |
| top10_interpretable | bad_core_nearboundary | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [0, 0, 119]] |
| top10_interpretable | bad_outlier_stress | 292 | 0.054795 | 0.034632 | 0.000000 | 0.000000 | 0.054795 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [276, 0, 16]] |
| top10_interpretable_badcal | original_test_all_10s+ | 8477 | 0.811962 | 0.714788 | 0.842308 | 0.830999 | 0.338200 | 0.750918 | 0.865005 | 0.978873 | 574 | 745 | 0 | [[3066, 574, 0], [745, 3678, 3], [272, 0, 139]] |
| top10_interpretable_badcal | original_all_10s+ | 32956 | 0.916798 | 0.924559 | 0.946899 | 0.853971 | 0.946074 | 0.898452 | 0.909692 | 0.996612 | 898 | 1542 | 3 | [[16138, 898, 7], [1542, 9076, 10], [282, 3, 5000]] |
| top10_interpretable_badcal | original_test_without_bad_outlier_stress | 8185 | 0.838485 | 0.886063 | 0.842308 | 0.830999 | 1.000000 | 0.804513 | 0.865005 | 0.975410 | 574 | 745 | 0 | [[3066, 574, 0], [745, 3678, 3], [0, 0, 119]] |
| top10_interpretable_badcal | bad_core_nearboundary | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [0, 0, 119]] |
| top10_interpretable_badcal | bad_outlier_stress | 292 | 0.068493 | 0.042735 | 0.000000 | 0.000000 | 0.068493 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [272, 0, 20]] |
| top14_balanced | original_test_all_10s+ | 8477 | 0.891707 | 0.891574 | 0.892308 | 0.898780 | 0.810219 | 0.875236 | 0.897766 | 0.994030 | 392 | 446 | 61 | [[3248, 392, 0], [446, 3978, 2], [17, 61, 333]] |
| top14_balanced | original_all_10s+ | 32956 | 0.965833 | 0.969621 | 0.971660 | 0.947027 | 0.984863 | 0.966330 | 0.948544 | 0.999424 | 483 | 560 | 63 | [[16560, 483, 0], [560, 10065, 3], [17, 63, 5205]] |
| top14_balanced | original_test_without_bad_outlier_stress | 8185 | 0.897373 | 0.927302 | 0.892308 | 0.898780 | 1.000000 | 0.879264 | 0.910297 | 0.983471 | 392 | 446 | 0 | [[3248, 392, 0], [446, 3978, 2], [0, 0, 119]] |
| top14_balanced | bad_core_nearboundary | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [0, 0, 119]] |
| top14_balanced | bad_outlier_stress | 292 | 0.732877 | 0.281950 | 0.000000 | 0.000000 | 0.732877 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 61 | [[0, 0, 0], [0, 0, 0], [17, 61, 214]] |
| top14_balanced_badcal | original_test_all_10s+ | 8477 | 0.898431 | 0.917485 | 0.892308 | 0.896521 | 0.973236 | 0.879502 | 0.909049 | 0.954654 | 392 | 439 | 5 | [[3248, 392, 0], [439, 3968, 19], [6, 5, 400]] |
| top14_balanced_badcal | original_all_10s+ | 32956 | 0.967593 | 0.972104 | 0.971660 | 0.946086 | 0.997729 | 0.967346 | 0.953623 | 0.996221 | 483 | 553 | 6 | [[16560, 483, 0], [553, 10055, 20], [6, 6, 5273]] |
| top14_balanced_badcal | original_test_without_bad_outlier_stress | 8185 | 0.896151 | 0.905303 | 0.892308 | 0.896521 | 1.000000 | 0.880933 | 0.910092 | 0.862319 | 392 | 439 | 0 | [[3248, 392, 0], [439, 3968, 19], [0, 0, 119]] |
| top14_balanced_badcal | bad_core_nearboundary | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [0, 0, 119]] |
| top14_balanced_badcal | bad_outlier_stress | 292 | 0.962329 | 0.326934 | 0.000000 | 0.000000 | 0.962329 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 5 | [[0, 0, 0], [0, 0, 0], [6, 5, 281]] |
| ranked_top18 | original_test_all_10s+ | 8477 | 0.796862 | 0.685885 | 0.926374 | 0.737460 | 0.289538 | 0.704114 | 0.917369 | 0.915385 | 268 | 1151 | 26 | [[3372, 268, 0], [1151, 3264, 11], [266, 26, 119]] |
| ranked_top18 | original_all_10s+ | 32956 | 0.880932 | 0.889420 | 0.927595 | 0.780109 | 0.933207 | 0.864116 | 0.863017 | 0.975861 | 1234 | 2215 | 82 | [[15809, 1234, 0], [2215, 8291, 122], [271, 82, 4932]] |
| ranked_top18 | original_test_without_bad_outlier_stress | 8185 | 0.825290 | 0.867432 | 0.926374 | 0.737460 | 1.000000 | 0.745523 | 0.924122 | 0.915385 | 268 | 1151 | 0 | [[3372, 268, 0], [1151, 3264, 11], [0, 0, 119]] |
| ranked_top18 | bad_core_nearboundary | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [0, 0, 119]] |
| ranked_top18 | bad_outlier_stress | 292 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 26 | [[0, 0, 0], [0, 0, 0], [266, 26, 0]] |
| ranked_top18_badcal | original_test_all_10s+ | 8477 | 0.796862 | 0.685885 | 0.926374 | 0.737460 | 0.289538 | 0.704114 | 0.917369 | 0.915385 | 268 | 1151 | 26 | [[3372, 268, 0], [1151, 3264, 11], [266, 26, 119]] |
| ranked_top18_badcal | original_all_10s+ | 32956 | 0.880932 | 0.889420 | 0.927595 | 0.780109 | 0.933207 | 0.864116 | 0.863017 | 0.975861 | 1234 | 2215 | 82 | [[15809, 1234, 0], [2215, 8291, 122], [271, 82, 4932]] |
| ranked_top18_badcal | original_test_without_bad_outlier_stress | 8185 | 0.825290 | 0.867432 | 0.926374 | 0.737460 | 1.000000 | 0.745523 | 0.924122 | 0.915385 | 268 | 1151 | 0 | [[3372, 268, 0], [1151, 3264, 11], [0, 0, 119]] |
| ranked_top18_badcal | bad_core_nearboundary | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | [[0, 0, 0], [0, 0, 0], [0, 0, 119]] |
| ranked_top18_badcal | bad_outlier_stress | 292 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 26 | [[0, 0, 0], [0, 0, 0], [266, 26, 0]] |

## Outputs

- Ranking CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\current_reduced_feature_ranking.csv`
- Selection CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\current_reduced_feature_trainval_selection.csv`
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\current_reduced_feature_metrics.csv`

## Notes

- No held-out test rows are used for feature ranking, model training, early stopping, or candidate selection.
- These are pure reduced-feature MLP candidates; no hand-written threshold/gate is added in v1.
