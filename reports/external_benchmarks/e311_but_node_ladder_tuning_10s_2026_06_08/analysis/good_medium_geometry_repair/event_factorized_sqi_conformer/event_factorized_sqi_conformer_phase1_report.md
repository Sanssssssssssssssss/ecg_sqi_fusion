# Event-Factorized SQI Conformer phase1 Report

- Generated: 2026-06-25 06:08:56
- Formal input contract: waveform-derived channels only.
- SQI/factor targets are training teacher/diagnostic targets only.
- All checkpoints in this stage are trained from scratch.

## Metrics

| candidate | bucket | n | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_precision | medium_precision | bad_precision | good_to_medium | medium_to_good | bad_to_medium | confusion_3x3 | macro_f1_sklearn | supported_labels | bad_fpr_nonbad | artifact_positive_nonbad_count | artifact_positive_nonbad_bad_fpr | factor_mae | quality_subtype_rows | quality_subtype_acc | quality_subtype_class_acc | boundary_four_rows | boundary_label_acc | boundary_family_acc | boundary_label_balanced_acc | record_macro_acc | record_macro_supported_f1 | record_macro_full_f1 | bad_record_count | bad_containing_record_bad_recall_mean | bad_containing_record_acc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | 4025 | 0.917516 | 0.899408 | 0.871006 | 0.860585 | 0.969772 | 0.85482 | 0.867303 | 0.973148 | 105 | 112 | 48 | [[736, 105, 4], [112, 1000, 50], [13, 48, 1957]] | 0.899408 | good,medium,bad | 0.0269058 | 838 | 0.0381862 | 1.99361 | 4025 | 0.512547 | 0.921491 | 99 | 0.909091 | 0.707071 | 0.910788 | 0.92407 | 0.924285 | 0.351384 | 1678 | 0.966826 | 0.960474 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | 4025 | 0.92 | 0.906501 | 0.872043 | 0.863951 | 0.976301 | 0.891209 | 0.879965 | 0.956105 | 107 | 82 | 29 | [[811, 107, 12], [82, 997, 75], [17, 29, 1895]] | 0.906501 | good,medium,bad | 0.0417466 | 828 | 0.0543478 | 1.81731 | 4025 | 0.474783 | 0.919503 | 113 | 0.911504 | 0.539823 | 0.908102 | 0.925364 | 0.924078 | 0.349362 | 1612 | 0.973428 | 0.97181 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | 4025 | 0.916025 | 0.900125 | 0.826882 | 0.898614 | 0.969088 | 0.901524 | 0.836965 | 0.973099 | 156 | 70 | 46 | [[769, 156, 5], [70, 1037, 47], [14, 46, 1881]] | 0.900125 | good,medium,bad | 0.024952 | 828 | 0.0362319 | 2.12937 | 4025 | 0.496149 | 0.917516 | 113 | 0.911504 | 0.513274 | 0.908102 | 0.923625 | 0.923538 | 0.348837 | 1612 | 0.965674 | 0.965627 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | 4025 | 0.92 | 0.905939 | 0.868817 | 0.877816 | 0.969603 | 0.881134 | 0.870275 | 0.968107 | 115 | 86 | 36 | [[808, 115, 7], [86, 1013, 55], [23, 36, 1882]] | 0.905939 | good,medium,bad | 0.0297505 | 828 | 0.0458937 | 1.84724 | 4025 | 0.48472 | 0.918261 | 113 | 0.911504 | 0.548673 | 0.905265 | 0.92493 | 0.923193 | 0.348815 | 1612 | 0.965674 | 0.966237 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | 4025 | 0.916522 | 0.897329 | 0.820118 | 0.879518 | 0.978196 | 0.890746 | 0.859546 | 0.959184 | 136 | 72 | 31 | [[693, 136, 16], [72, 1022, 68], [13, 31, 1974]] | 0.897329 | good,medium,bad | 0.0418535 | 838 | 0.0548926 | 2.03229 | 4025 | 0.508075 | 0.917516 | 99 | 0.89899 | 0.676768 | 0.895611 | 0.922256 | 0.923052 | 0.35145 | 1678 | 0.976957 | 0.970461 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | 4025 | 0.914286 | 0.893886 | 0.811834 | 0.876936 | 0.978692 | 0.886305 | 0.85272 | 0.960603 | 140 | 81 | 36 | [[686, 140, 19], [81, 1019, 62], [7, 36, 1975]] | 0.893886 | good,medium,bad | 0.0403587 | 838 | 0.0536993 | 1.92988 | 4025 | 0.493168 | 0.912795 | 99 | 0.919192 | 0.666667 | 0.91735 | 0.921191 | 0.921061 | 0.350109 | 1678 | 0.975963 | 0.967541 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | 4024 | 0.914761 | 0.894723 | 0.877381 | 0.846019 | 0.968643 | 0.82716 | 0.878292 | 0.972933 | 99 | 125 | 35 | [[737, 99, 4], [125, 967, 51], [29, 35, 1977]] | 0.894723 | good,medium,bad | 0.0277358 | 825 | 0.0436364 | 1.94422 | 4024 | 0.510686 | 0.916998 | 105 | 0.904762 | 0.733333 | 0.902198 | 0.921221 | 0.920631 | 0.346346 | 1676 | 0.965891 | 0.959755 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | 4024 | 0.915507 | 0.896048 | 0.85119 | 0.866142 | 0.969623 | 0.863527 | 0.854922 | 0.97105 | 120 | 99 | 48 | [[715, 120, 5], [99, 990, 54], [14, 48, 1979]] | 0.896048 | good,medium,bad | 0.0297529 | 825 | 0.0472727 | 1.9129 | 4024 | 0.510934 | 0.920477 | 105 | 0.933333 | 0.609524 | 0.936289 | 0.918619 | 0.917783 | 0.346031 | 1676 | 0.967383 | 0.960734 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | 4024 | 0.911034 | 0.890476 | 0.85119 | 0.864392 | 0.961783 | 0.841176 | 0.846615 | 0.978077 | 119 | 117 | 60 | [[715, 119, 6], [117, 988, 38], [18, 60, 1963]] | 0.890476 | good,medium,bad | 0.0221886 | 825 | 0.0412121 | 1.88238 | 4024 | 0.509443 | 0.912276 | 105 | 0.92381 | 0.752381 | 0.928092 | 0.915608 | 0.913602 | 0.34359 | 1676 | 0.958433 | 0.955459 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_val | 1631 | 0.921521 | 0.903188 | 0.87027 | 0.853933 | 0.981618 | 0.875 | 0.86758 | 0.970909 | 46 | 43 | 12 | [[322, 46, 2], [43, 380, 22], [3, 12, 801]] | 0.903188 | good,medium,bad | 0.0294479 | 350 | 0.0342857 | 2.00129 | 1631 | 0.506438 | 0.918455 | 43 | 0.860465 | 0.651163 | 0.864253 | 0.929678 | 0.929275 | 0.354915 | 676 | 0.981509 | 0.974926 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_val | 1626 | 0.926199 | 0.912307 | 0.894895 | 0.895178 | 0.957108 | 0.871345 | 0.875 | 0.981156 | 33 | 37 | 28 | [[298, 33, 2], [37, 427, 13], [7, 28, 781]] | 0.912307 | good,medium,bad | 0.0185185 | 325 | 0.0338462 | 1.8745 | 1626 | 0.50738 | 0.926814 | 47 | 0.914894 | 0.787234 | 0.920113 | 0.932175 | 0.928559 | 0.355318 | 676 | 0.952416 | 0.957964 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_val | 1626 | 0.922509 | 0.907459 | 0.873874 | 0.8826 | 0.965686 | 0.881818 | 0.871636 | 0.96925 | 40 | 33 | 22 | [[291, 40, 2], [33, 421, 23], [6, 22, 788]] | 0.907459 | good,medium,bad | 0.0308642 | 325 | 0.0584615 | 1.90401 | 1626 | 0.49631 | 0.926199 | 47 | 0.914894 | 0.744681 | 0.920113 | 0.925647 | 0.923468 | 0.354532 | 676 | 0.961045 | 0.962747 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_val | 1631 | 0.912324 | 0.89114 | 0.813514 | 0.853933 | 0.988971 | 0.890533 | 0.850112 | 0.953901 | 61 | 34 | 6 | [[301, 61, 8], [34, 380, 31], [3, 6, 807]] | 0.89114 | good,medium,bad | 0.0478528 | 350 | 0.0542857 | 1.93021 | 1631 | 0.491723 | 0.909871 | 43 | 0.906977 | 0.627907 | 0.892534 | 0.921901 | 0.920539 | 0.350625 | 676 | 0.988905 | 0.980005 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_val | 1626 | 0.917589 | 0.89918 | 0.87988 | 0.855346 | 0.969363 | 0.837143 | 0.883117 | 0.971744 | 38 | 48 | 16 | [[293, 38, 2], [48, 408, 21], [9, 16, 791]] | 0.89918 | good,medium,bad | 0.0283951 | 325 | 0.0523077 | 1.93808 | 1626 | 0.495695 | 0.916974 | 47 | 0.851064 | 0.702128 | 0.841165 | 0.921676 | 0.91934 | 0.352586 | 676 | 0.966223 | 0.96467 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_val | 1637 | 0.912034 | 0.892136 | 0.840841 | 0.865424 | 0.968331 | 0.851064 | 0.860082 | 0.967153 | 50 | 41 | 18 | [[280, 50, 3], [41, 418, 24], [8, 18, 795]] | 0.892136 | good,medium,bad | 0.0330882 | 328 | 0.027439 | 1.84432 | 1637 | 0.484423 | 0.90898 | 45 | 0.888889 | 0.555556 | 0.89916 | 0.917554 | 0.91832 | 0.351357 | 677 | 0.964549 | 0.960463 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_val | 1631 | 0.910484 | 0.887992 | 0.813514 | 0.849438 | 0.987745 | 0.877551 | 0.841871 | 0.960667 | 65 | 38 | 6 | [[301, 65, 4], [38, 378, 29], [4, 6, 806]] | 0.887992 | good,medium,bad | 0.0404908 | 350 | 0.0457143 | 2.03788 | 1631 | 0.496015 | 0.910484 | 43 | 0.883721 | 0.651163 | 0.873303 | 0.919555 | 0.917415 | 0.348787 | 676 | 0.987426 | 0.977712 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_val | 1637 | 0.909591 | 0.888816 | 0.816817 | 0.871636 | 0.969549 | 0.86901 | 0.842 | 0.966019 | 58 | 37 | 21 | [[272, 58, 3], [37, 421, 25], [4, 21, 796]] | 0.888816 | good,medium,bad | 0.0343137 | 328 | 0.0243902 | 2.1258 | 1637 | 0.516188 | 0.90898 | 45 | 0.911111 | 0.511111 | 0.917017 | 0.914997 | 0.914958 | 0.348859 | 677 | 0.965288 | 0.958124 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_val | 1637 | 0.907147 | 0.886291 | 0.828829 | 0.846791 | 0.974421 | 0.854489 | 0.859244 | 0.954654 | 51 | 42 | 16 | [[276, 51, 6], [42, 409, 32], [5, 16, 800]] | 0.886291 | good,medium,bad | 0.0465686 | 328 | 0.0396341 | 1.82051 | 1637 | 0.471594 | 0.907758 | 45 | 0.911111 | 0.577778 | 0.928571 | 0.912981 | 0.913335 | 0.349515 | 677 | 0.971196 | 0.962334 |

## Feature Recovery

| candidate | bucket | feature | corr_all | mae | corr_good | corr_medium | corr_bad | corr_min_supported_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | qrs_visibility | 0.366158 | 3.64202 | nan | 0.199123 | 0.266821 | 0.199123 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | detector_agreement | 0.81955 | 0.324358 | 0.367176 | 0.38775 | 0.741739 | 0.367176 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | baseline_step | 0.940083 | 0.252886 | 0.920916 | 0.843746 | 0.60513 | 0.60513 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | flatline_ratio | 0.790746 | 3.12445 | 0.710775 | 0.592596 | 0.804294 | 0.592596 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | sqi_basSQI | 0.585573 | 0.574892 | 0.30512 | -0.329226 | 0.368801 | -0.329226 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | non_qrs_diff_p95 | 0.884774 | 0.355534 | 0.784512 | 0.857939 | 0.948268 | 0.784512 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | non_qrs_rms_ratio | 0.605326 | 1.6974 | 0.507542 | -0.144808 | 0.712749 | -0.144808 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | qrs_band_ratio | 0 | 3.94583 | nan | nan | nan | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | template_corr | 0.840874 | 1.29016 | 0.220034 | 0.198174 | 0.626067 | 0.198174 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | amplitude_entropy | 0.251772 | 0.596666 | 0.222882 | 0.231157 | 0.264908 | 0.222882 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | contact_loss_win_ratio | 0.286654 | 5.42448 | 0.0684776 | 0.329961 | 0.394347 | 0.0684776 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | qrs_visibility | 0.3595 | 3.82595 | nan | 0.161426 | 0.260092 | 0.161426 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | detector_agreement | 0.820135 | 0.316756 | 0.376835 | 0.346768 | 0.748337 | 0.346768 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | baseline_step | 0.95981 | 0.183257 | 0.925183 | 0.894217 | 0.765689 | 0.765689 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | flatline_ratio | 0.786302 | 3.44064 | 0.701542 | 0.56296 | 0.798158 | 0.56296 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | sqi_basSQI | 0.758298 | 0.620176 | 0.673047 | 0.115935 | 0.434073 | 0.115935 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | non_qrs_diff_p95 | 0.892186 | 0.333715 | 0.806701 | 0.877928 | 0.951119 | 0.806701 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | non_qrs_rms_ratio | 0.628146 | 1.92131 | 0.58725 | 0.161401 | 0.602223 | 0.161401 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | qrs_band_ratio | 0 | 3.9752 | nan | nan | nan | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | template_corr | 0.849397 | 1.19659 | 0.210478 | 0.209622 | 0.567118 | 0.209622 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | amplitude_entropy | 0.256714 | 0.593933 | 0.329656 | 0.192018 | 0.136023 | 0.136023 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | contact_loss_win_ratio | 0.236174 | 5.94764 | 0.0861915 | 0.245383 | 0.34192 | 0.0861915 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | qrs_visibility | 0.371038 | 3.54803 | nan | 0.212487 | 0.237962 | 0.212487 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | detector_agreement | 0.82353 | 0.324989 | 0.365199 | 0.35074 | 0.763827 | 0.35074 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | baseline_step | 0.939863 | 0.240902 | 0.906477 | 0.850937 | 0.706202 | 0.706202 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | flatline_ratio | 0.796996 | 3.17298 | 0.698039 | 0.564742 | 0.820226 | 0.564742 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | sqi_basSQI | 0.667929 | 0.574707 | 0.656904 | -0.153877 | 0.449944 | -0.153877 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | non_qrs_diff_p95 | 0.880336 | 0.380923 | 0.800089 | 0.869689 | 0.932782 | 0.800089 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | non_qrs_rms_ratio | 0.639687 | 1.79347 | 0.677875 | -0.0631107 | 0.700066 | -0.0631107 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | qrs_band_ratio | 0 | 4.34344 | nan | nan | nan | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | template_corr | 0.856112 | 1.24726 | 0.280322 | 0.213317 | 0.659808 | 0.213317 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | amplitude_entropy | 0.26902 | 0.582844 | 0.35791 | 0.157924 | 0.211306 | 0.157924 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | contact_loss_win_ratio | 0.219532 | 5.72016 | -0.00339972 | 0.247813 | 0.372469 | -0.00339972 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | qrs_visibility | 0.385039 | 3.51805 | nan | 0.203993 | 0.204074 | 0.203993 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | detector_agreement | 0.807556 | 0.322943 | 0.210234 | 0.341176 | 0.787336 | 0.210234 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | baseline_step | 0.919905 | 0.311256 | 0.862011 | 0.856284 | 0.651558 | 0.651558 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | flatline_ratio | 0.790683 | 2.88091 | 0.656777 | 0.575249 | 0.817707 | 0.575249 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | sqi_basSQI | 0.571747 | 0.580965 | 0.251546 | -0.436466 | 0.378945 | -0.436466 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | non_qrs_diff_p95 | 0.884224 | 0.2995 | 0.757677 | 0.871321 | 0.945965 | 0.757677 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | non_qrs_rms_ratio | 0.608978 | 1.52182 | 0.57947 | -0.0737279 | 0.760022 | -0.0737279 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | qrs_band_ratio | 0 | 3.40677 | nan | nan | nan | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | template_corr | 0.807305 | 1.43225 | 0.221917 | 0.208082 | 0.560952 | 0.208082 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | amplitude_entropy | 0.216175 | 0.560268 | 0.269928 | 0.0794145 | 0.263445 | 0.0794145 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | contact_loss_win_ratio | 0.188157 | 5.48489 | -0.0194711 | 0.342676 | 0.358481 | -0.0194711 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | qrs_visibility | 0.395732 | 3.97258 | nan | 0.171409 | 0.245674 | 0.171409 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | detector_agreement | 0.803827 | 0.317857 | 0.142547 | 0.265238 | 0.764797 | 0.142547 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | baseline_step | 0.943148 | 0.21796 | 0.922994 | 0.892249 | 0.67404 | 0.67404 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | flatline_ratio | 0.813037 | 3.19444 | 0.768175 | 0.623118 | 0.815155 | 0.623118 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | sqi_basSQI | 0.606665 | 0.564647 | 0.323902 | -0.310683 | 0.404555 | -0.310683 |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | clean_test | non_qrs_diff_p95 | 0.903272 | 0.288326 | 0.807675 | 0.88847 | 0.952746 | 0.807675 |

_Showing 50 of 80 rows._

## Record Metrics Preview

| candidate | record_id | rows | good_rows | medium_rows | bad_rows | full_macro_f1 | supported_macro_f1 | acc | good_recall | medium_recall | bad_recall | artifact_positive_nonbad | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10035 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10071 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10101 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10133 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10161 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10176 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10178 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10179 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_102 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10301 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10353 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10468 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10482 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10498 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10546 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10627 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10633 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10635 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10727 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_10886 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_1090 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_1102 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11071 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_1108 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11128 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11130 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11135 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11141 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11207 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_1121 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11273 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11327 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11377 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11433 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11438 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11459 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11481 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11556 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11586 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11592 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11608 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11688 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11777 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11780 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11845 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11847 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11910 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11962 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_11977 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| E4_query_highres_local_art_unified_lowaux_lr15e4 | v108_ptb_ptbxl_12092 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |

_Showing 50 of 80 rows._

## Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_feature_recovery.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_record_metrics.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_sqi_conformer`