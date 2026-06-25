# Event-Factorized SQI Conformer phase3 Report

- Generated: 2026-06-25 05:10:23
- Formal input contract: waveform-derived channels only.
- SQI/factor targets are training teacher/diagnostic targets only.
- All checkpoints in this stage are trained from scratch.

## Metrics

| candidate | bucket | n | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_precision | medium_precision | bad_precision | good_to_medium | medium_to_good | bad_to_medium | confusion_3x3 | macro_f1_sklearn | supported_labels | bad_fpr_nonbad | artifact_positive_nonbad_count | artifact_positive_nonbad_bad_fpr | factor_mae | quality_subtype_rows | quality_subtype_acc | quality_subtype_class_acc | boundary_four_rows | boundary_label_acc | boundary_family_acc | boundary_label_balanced_acc | record_macro_acc | record_macro_supported_f1 | record_macro_full_f1 | bad_record_count | bad_containing_record_bad_recall_mean | bad_containing_record_acc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | 4025 | 0.928944 | 0.917018 | 0.931183 | 0.863085 | 0.967027 | 0.853202 | 0.917127 | 0.975572 | 56 | 119 | 34 | [[866, 56, 8], [119, 996, 39], [30, 34, 1877]] | 0.917018 | good,medium,bad | 0.0225528 | 828 | 0.0386473 | 2.0533 | 4025 | 0.51205 | 0.929441 | 113 | 0.920354 | 0.566372 | 0.919136 | 0.933974 | 0.932375 | 0.352643 | 1612 | 0.963193 | 0.964376 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | 4025 | 0.924472 | 0.908158 | 0.878107 | 0.856282 | 0.983152 | 0.870892 | 0.906193 | 0.956145 | 83 | 96 | 20 | [[742, 83, 20], [96, 995, 71], [14, 20, 1984]] | 0.908158 | good,medium,bad | 0.0453413 | 838 | 0.0548926 | 1.96481 | 4025 | 0.524224 | 0.920994 | 99 | 0.919192 | 0.757576 | 0.91735 | 0.931763 | 0.931929 | 0.354408 | 1678 | 0.981724 | 0.97354 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | 4025 | 0.919752 | 0.902059 | 0.850888 | 0.898451 | 0.960852 | 0.86731 | 0.853639 | 0.982767 | 123 | 87 | 56 | [[719, 123, 3], [87, 1044, 31], [23, 56, 1939]] | 0.902059 | good,medium,bad | 0.0169407 | 838 | 0.0274463 | 2.05958 | 4025 | 0.529938 | 0.922733 | 99 | 0.909091 | 0.636364 | 0.910788 | 0.926781 | 0.926182 | 0.351658 | 1678 | 0.95739 | 0.953327 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | 4025 | 0.921242 | 0.907071 | 0.953763 | 0.817158 | 0.967543 | 0.815257 | 0.930898 | 0.976091 | 37 | 171 | 33 | [[887, 37, 6], [171, 943, 40], [30, 33, 1878]] | 0.907071 | good,medium,bad | 0.0220729 | 828 | 0.0410628 | 2.05403 | 4025 | 0.503106 | 0.924224 | 113 | 0.893805 | 0.610619 | 0.894546 | 0.928547 | 0.925646 | 0.34899 | 1612 | 0.963503 | 0.963942 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | 4024 | 0.917992 | 0.899783 | 0.896429 | 0.851269 | 0.964233 | 0.832044 | 0.880543 | 0.97716 | 84 | 127 | 48 | [[753, 84, 3], [127, 973, 43], [25, 48, 1968]] | 0.899783 | good,medium,bad | 0.0231972 | 825 | 0.04 | 1.97074 | 4024 | 0.542992 | 0.920974 | 105 | 0.92381 | 0.609524 | 0.928092 | 0.922175 | 0.921526 | 0.347286 | 1676 | 0.960521 | 0.955315 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | 4024 | 0.914513 | 0.894313 | 0.838095 | 0.873141 | 0.969133 | 0.866995 | 0.84648 | 0.972946 | 134 | 92 | 47 | [[704, 134, 2], [92, 998, 53], [16, 47, 1978]] | 0.894313 | good,medium,bad | 0.0277358 | 825 | 0.0412121 | 1.92538 | 4024 | 0.539016 | 0.919235 | 105 | 0.885714 | 0.657143 | 0.879471 | 0.919452 | 0.918508 | 0.345959 | 1676 | 0.966388 | 0.960679 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_val | 1631 | 0.923973 | 0.907166 | 0.881081 | 0.840449 | 0.988971 | 0.88587 | 0.892601 | 0.956161 | 39 | 39 | 6 | [[326, 39, 5], [39, 374, 32], [3, 6, 807]] | 0.907166 | good,medium,bad | 0.0453988 | 350 | 0.04 | 1.96907 | 1631 | 0.52851 | 0.92336 | 43 | 0.930233 | 0.744186 | 0.921946 | 0.93198 | 0.931615 | 0.356147 | 676 | 0.988905 | 0.980621 |
| P2_ecg_mask_unified_lowaux_v112 | clean_val | 1637 | 0.918754 | 0.900848 | 0.900901 | 0.84058 | 0.971985 | 0.831025 | 0.894273 | 0.970803 | 32 | 54 | 16 | [[300, 32, 1], [54, 406, 23], [7, 16, 798]] | 0.900848 | good,medium,bad | 0.0294118 | 328 | 0.0304878 | 2.05415 | 1637 | 0.515577 | 0.916921 | 45 | 0.888889 | 0.6 | 0.910714 | 0.929452 | 0.929665 | 0.35498 | 677 | 0.968242 | 0.962826 |
| P2_ecg_mask_unified_lowaux_v112 | clean_val | 1631 | 0.918455 | 0.897826 | 0.843243 | 0.869663 | 0.979167 | 0.869081 | 0.844978 | 0.981572 | 57 | 44 | 14 | [[312, 57, 1], [44, 387, 14], [3, 14, 799]] | 0.897826 | good,medium,bad | 0.0184049 | 350 | 0.0342857 | 2.0641 | 1631 | 0.522992 | 0.917842 | 43 | 0.930233 | 0.72093 | 0.921946 | 0.92861 | 0.927183 | 0.35372 | 676 | 0.977811 | 0.972485 |
| P2_ecg_mask_unified_lowaux_v112 | clean_val | 1626 | 0.922509 | 0.906486 | 0.858859 | 0.895178 | 0.964461 | 0.885449 | 0.860887 | 0.975217 | 45 | 32 | 24 | [[286, 45, 2], [32, 427, 18], [5, 24, 787]] | 0.906486 | good,medium,bad | 0.0246914 | 325 | 0.0430769 | 1.91725 | 1626 | 0.519065 | 0.926199 | 47 | 0.893617 | 0.744681 | 0.885338 | 0.926414 | 0.924143 | 0.354836 | 676 | 0.961045 | 0.964053 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_val | 1626 | 0.921279 | 0.906477 | 0.897898 | 0.880503 | 0.954657 | 0.851852 | 0.876827 | 0.978643 | 32 | 42 | 27 | [[299, 32, 2], [42, 420, 15], [10, 27, 779]] | 0.906477 | good,medium,bad | 0.0209877 | 325 | 0.0369231 | 1.96322 | 1626 | 0.51845 | 0.926199 | 47 | 0.914894 | 0.723404 | 0.911654 | 0.925677 | 0.923802 | 0.354782 | 676 | 0.950197 | 0.953107 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_val | 1637 | 0.913256 | 0.895048 | 0.93994 | 0.807453 | 0.964677 | 0.794416 | 0.909091 | 0.972973 | 19 | 72 | 20 | [[313, 19, 1], [72, 390, 21], [9, 20, 792]] | 0.895048 | good,medium,bad | 0.0269608 | 328 | 0.0335366 | 2.05368 | 1637 | 0.501527 | 0.912645 | 45 | 0.911111 | 0.577778 | 0.928571 | 0.922699 | 0.922219 | 0.352022 | 677 | 0.95938 | 0.954234 |

## Feature Recovery

| candidate | bucket | feature | corr_all | mae | corr_good | corr_medium | corr_bad | corr_min_supported_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | qrs_visibility | 0.375527 | 4.14914 | nan | 0.184548 | 0.221383 | 0.184548 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | detector_agreement | 0.813376 | 0.30148 | 0.28655 | 0.227165 | 0.770757 | 0.227165 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | baseline_step | 0.926689 | 0.239759 | 0.914914 | 0.802915 | 0.666541 | 0.666541 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | flatline_ratio | 0.808731 | 3.15482 | 0.693761 | 0.610557 | 0.833758 | 0.610557 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | sqi_basSQI | 0.560253 | 0.596937 | 0.289891 | -0.310565 | 0.38841 | -0.310565 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | non_qrs_diff_p95 | 0.92077 | 0.260723 | 0.843876 | 0.90088 | 0.977854 | 0.843876 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | non_qrs_rms_ratio | 0.599099 | 1.516 | 0.575054 | -0.210987 | 0.648189 | -0.210987 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | qrs_band_ratio | 0 | 4.21919 | nan | nan | nan | nan |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | template_corr | 0.873858 | 1.24445 | 0.237111 | 0.259643 | 0.72004 | 0.237111 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | amplitude_entropy | 0.284728 | 0.612626 | 0.413331 | 0.189432 | 0.222029 | 0.189432 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | contact_loss_win_ratio | 0.235123 | 6.36025 | 0.0390066 | 0.222368 | 0.322227 | 0.0390066 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | qrs_visibility | 0.387027 | 4.11184 | nan | 0.167553 | 0.199822 | 0.167553 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | detector_agreement | 0.818967 | 0.295283 | 0.343625 | 0.411265 | 0.775739 | 0.343625 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | baseline_step | 0.959505 | 0.232753 | 0.932966 | 0.903234 | 0.787437 | 0.787437 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | flatline_ratio | 0.772868 | 3.06534 | 0.674127 | 0.604238 | 0.769583 | 0.604238 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | sqi_basSQI | 0.6888 | 0.490715 | 0.816348 | -0.010812 | 0.365113 | -0.010812 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | non_qrs_diff_p95 | 0.900319 | 0.285938 | 0.839165 | 0.901476 | 0.962388 | 0.839165 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | non_qrs_rms_ratio | 0.635958 | 1.40533 | 0.686089 | -0.105418 | 0.682544 | -0.105418 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | qrs_band_ratio | 0 | 4.89019 | nan | nan | nan | nan |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | template_corr | 0.87067 | 1.2864 | 0.224537 | 0.258974 | 0.69995 | 0.224537 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | amplitude_entropy | 0.39136 | 0.561001 | 0.509981 | 0.315511 | 0.251155 | 0.251155 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | contact_loss_win_ratio | 0.181525 | 5.96147 | 0.0017696 | 0.233032 | 0.334125 | 0.0017696 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | qrs_visibility | 0.388723 | 3.55269 | nan | 0.344044 | 0.218559 | 0.218559 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | detector_agreement | 0.819153 | 0.292049 | 0.289831 | 0.264517 | 0.787831 | 0.264517 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | baseline_step | 0.952861 | 0.211507 | 0.926739 | 0.866248 | 0.678982 | 0.678982 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | flatline_ratio | 0.793381 | 3.22514 | 0.726073 | 0.651398 | 0.815764 | 0.651398 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | sqi_basSQI | 0.649711 | 0.551052 | 0.536598 | -0.270176 | 0.34511 | -0.270176 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | non_qrs_diff_p95 | 0.907097 | 0.299691 | 0.820692 | 0.899024 | 0.965302 | 0.820692 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | non_qrs_rms_ratio | 0.668114 | 1.52554 | 0.592508 | 0.20023 | 0.753886 | 0.20023 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | qrs_band_ratio | 0 | 3.85493 | nan | nan | nan | nan |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | template_corr | 0.870473 | 1.30684 | 0.160465 | 0.299593 | 0.709329 | 0.160465 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | amplitude_entropy | 0.406854 | 0.547024 | 0.539529 | 0.456575 | 0.208961 | 0.208961 |
| P2_ecg_mask_unified_lowaux_v112 | clean_test | contact_loss_win_ratio | 0.210694 | 5.81269 | 0.0763978 | 0.318023 | 0.235961 | 0.0763978 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | qrs_visibility | 0.37586 | 3.89812 | nan | 0.184716 | 0.191642 | 0.184716 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | detector_agreement | 0.810236 | 0.299767 | 0.335019 | 0.255552 | 0.755398 | 0.255552 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | baseline_step | 0.938285 | 0.222416 | 0.920908 | 0.849978 | 0.660405 | 0.660405 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | flatline_ratio | 0.795757 | 3.11735 | 0.699455 | 0.581833 | 0.826243 | 0.581833 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | sqi_basSQI | 0.638244 | 0.594521 | 0.48493 | -0.175113 | 0.400124 | -0.175113 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | non_qrs_diff_p95 | 0.919475 | 0.265669 | 0.83458 | 0.900809 | 0.973972 | 0.83458 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | non_qrs_rms_ratio | 0.609114 | 1.56779 | 0.622233 | -0.0581476 | 0.62032 | -0.0581476 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | qrs_band_ratio | 0 | 3.81493 | nan | nan | nan | nan |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | template_corr | 0.863132 | 1.27094 | 0.24958 | 0.215532 | 0.685168 | 0.215532 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | amplitude_entropy | 0.349342 | 0.563699 | 0.403047 | 0.391879 | 0.156748 | 0.156748 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | contact_loss_win_ratio | 0.198463 | 5.99773 | 0.0483615 | 0.221675 | 0.250276 | 0.0483615 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | qrs_visibility | 0.379675 | 4.04623 | nan | 0.162776 | 0.190554 | 0.162776 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | detector_agreement | 0.829158 | 0.297708 | 0.436512 | 0.463516 | 0.783263 | 0.436512 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | baseline_step | 0.960618 | 0.324344 | 0.946369 | 0.907748 | 0.77061 | 0.77061 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | flatline_ratio | 0.781054 | 3.03137 | 0.683943 | 0.579631 | 0.78176 | 0.579631 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | sqi_basSQI | 0.602141 | 0.485292 | 0.728634 | -0.245464 | 0.36009 | -0.245464 |
| P3_factorized_proxy_unified_gmweighted_v112 | clean_test | non_qrs_diff_p95 | 0.913501 | 0.283487 | 0.842654 | 0.898598 | 0.964645 | 0.842654 |

_Showing 50 of 66 rows._

## Record Metrics Preview

| candidate | record_id | rows | good_rows | medium_rows | bad_rows | full_macro_f1 | supported_macro_f1 | acc | good_recall | medium_recall | bad_recall | artifact_positive_nonbad | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10035 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10071 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10101 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10133 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10161 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10176 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10178 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10179 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_102 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10301 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10353 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10468 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10482 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10498 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10546 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10627 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10633 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10635 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10727 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_10886 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_1090 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_1102 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11071 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_1108 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11128 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11130 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11135 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11141 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11207 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_1121 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11273 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11327 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11377 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11433 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11438 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11459 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11481 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11556 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11586 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11592 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11608 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11688 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11777 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11780 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11845 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11847 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11910 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11962 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_11977 | 2 | 0 | 0 | 2 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |
| P2_ecg_mask_unified_lowaux_v112 | v108_ptb_ptbxl_12092 | 1 | 0 | 0 | 1 | 0.333333 | 1 | 1 | 0 | 0 | 1 | 0 | nan |

_Showing 50 of 80 rows._

## Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase3_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase3_feature_recovery.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase3_record_metrics.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_sqi_conformer`