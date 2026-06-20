# Event-Factorized SQI Conformer phase1 Report

- Generated: 2026-06-20 05:45:06
- Formal input contract: waveform-derived channels only.
- SQI/factor targets are training teacher/diagnostic targets only.
- All checkpoints in this stage are trained from scratch.

## Metrics

| candidate | bucket | n | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_precision | medium_precision | bad_precision | good_to_medium | medium_to_good | bad_to_medium | confusion_3x3 | macro_f1_sklearn | supported_labels | bad_fpr_nonbad | artifact_positive_nonbad_count | artifact_positive_nonbad_bad_fpr | factor_mae | record_macro_acc | record_macro_supported_f1 | record_macro_full_f1 | bad_record_count | bad_containing_record_bad_recall_mean | bad_containing_record_acc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_query_only | clean_test | 853 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | [[656, 0, 0], [0, 79, 0], [0, 0, 118]] | 1 | good,medium,bad | 0 | 104 | 0 | 1.15713 | 1 | 1 | 0.583333 | 1 | 1 | 1 |
| E3_query_highres_local | clean_test | 853 | 0.998828 | 0.997649 | 0.998476 | 1 | 1 | 1 | 0.9875 | 1 | 1 | 0 | 0 | [[655, 1, 0], [0, 79, 0], [0, 0, 118]] | 0.997649 | good,medium,bad | 0 | 104 | 0 | 1.2056 | 0.982143 | 0.990385 | 0.580128 | 1 | 1 | 1 |
| E4_query_highres_local_art | clean_test | 853 | 0.995311 | 0.99075 | 0.993902 | 1 | 1 | 1 | 0.951807 | 1 | 4 | 0 | 0 | [[652, 4, 0], [0, 79, 0], [0, 0, 118]] | 0.99075 | good,medium,bad | 0 | 104 | 0 | 1.19257 | 0.978571 | 0.988573 | 0.579524 | 1 | 1 | 1 |
| E0_noquery_nohi_nolocal_noart | clean_test | 853 | 0.996483 | 0.990686 | 0.998476 | 1 | 0.983051 | 1 | 0.963415 | 1 | 1 | 0 | 2 | [[655, 1, 0], [0, 79, 0], [0, 2, 116]] | 0.990686 | good,medium,bad | 0 | 104 | 0 | 1.20333 | 0.980964 | 0.988362 | 0.578105 | 1 | 0.983051 | 0.990566 |
| E2_query_highres | clean_test | 853 | 0.994138 | 0.9885 | 0.992378 | 1 | 1 | 1 | 0.940476 | 1 | 5 | 0 | 0 | [[651, 5, 0], [0, 79, 0], [0, 0, 118]] | 0.9885 | good,medium,bad | 0 | 104 | 0 | 1.2819 | 0.977381 | 0.987957 | 0.579319 | 1 | 1 | 1 |
| E2_query_highres | clean_test | 853 | 0.991794 | 0.984071 | 0.989329 | 1 | 1 | 1 | 0.918605 | 1 | 7 | 0 | 0 | [[649, 7, 0], [0, 79, 0], [0, 0, 118]] | 0.984071 | good,medium,bad | 0 | 104 | 0 | 1.16838 | 0.975 | 0.986708 | 0.578903 | 1 | 1 | 1 |
| E0_noquery_nohi_nolocal_noart | clean_test | 853 | 0.995311 | 0.990438 | 0.998476 | 0.962025 | 1 | 0.995441 | 0.987013 | 1 | 1 | 3 | 0 | [[655, 1, 0], [3, 76, 0], [0, 0, 118]] | 0.990438 | good,medium,bad | 0 | 104 | 0 | 1.00826 | 0.979829 | 0.986436 | 0.577102 | 1 | 1 | 0.995283 |
| E1_query_only | clean_test | 853 | 0.995311 | 0.990438 | 0.998476 | 0.962025 | 1 | 0.995441 | 0.987013 | 1 | 1 | 3 | 0 | [[655, 1, 0], [3, 76, 0], [0, 0, 118]] | 0.990438 | good,medium,bad | 0 | 104 | 0 | 0.928267 | 0.979829 | 0.986436 | 0.577102 | 1 | 1 | 0.995283 |
| E3_query_highres_local | clean_test | 853 | 0.995311 | 0.990438 | 0.998476 | 0.962025 | 1 | 0.995441 | 0.987013 | 1 | 1 | 3 | 0 | [[655, 1, 0], [3, 76, 0], [0, 0, 118]] | 0.990438 | good,medium,bad | 0 | 104 | 0 | 1.00899 | 0.979829 | 0.986436 | 0.577102 | 1 | 1 | 0.995283 |
| E2_query_highres | clean_test | 853 | 0.995311 | 0.990649 | 0.995427 | 0.987342 | 1 | 0.998471 | 0.962963 | 1 | 3 | 1 | 0 | [[653, 3, 0], [1, 78, 0], [0, 0, 118]] | 0.990649 | good,medium,bad | 0 | 104 | 0 | 1.02378 | 0.979004 | 0.975755 | 0.570181 | 1 | 1 | 0.995283 |
| E1_query_only | clean_test | 10767 | 0.97901 | 0.974438 | 0.995169 | 0.909333 | 1 | 0.957128 | 0.989362 | 0.999748 | 22 | 203 | 0 | [[4532, 22, 0], [203, 2046, 1], [0, 0, 3963]] | 0.974438 | good,medium,bad | 0.000146972 | 2223 | 0.000449843 | 1.46956 | 0.97901 | 0.974438 | 0.974438 | 1 | 1 | 0.97901 |
| E3_query_highres_local | clean_test | 10767 | 0.978638 | 0.973998 | 0.994291 | 0.909333 | 1 | 0.957092 | 0.987452 | 0.999748 | 26 | 203 | 0 | [[4528, 26, 0], [203, 2046, 1], [0, 0, 3963]] | 0.973998 | good,medium,bad | 0.000146972 | 2223 | 0.000449843 | 1.51545 | 0.978638 | 0.973998 | 0.973998 | 1 | 1 | 0.978638 |
| E3_query_highres_local | clean_test | 853 | 0.992966 | 0.985973 | 0.993902 | 0.974684 | 1 | 0.996942 | 0.950617 | 1 | 4 | 2 | 0 | [[652, 4, 0], [2, 77, 0], [0, 0, 118]] | 0.985973 | good,medium,bad | 0 | 104 | 0 | 1.0327 | 0.976952 | 0.973787 | 0.56907 | 1 | 1 | 0.995283 |
| E3_query_highres_local | clean_test | 10767 | 0.97836 | 0.973624 | 0.995389 | 0.905778 | 1 | 0.955321 | 0.989801 | 1 | 21 | 212 | 0 | [[4533, 21, 0], [212, 2038, 0], [0, 0, 3963]] | 0.973624 | good,medium,bad | 0 | 2223 | 0 | 1.51293 | 0.97836 | 0.973624 | 0.973624 | 1 | 1 | 0.97836 |
| E2_query_highres | clean_test | 10767 | 0.976502 | 0.971288 | 0.995169 | 0.897333 | 1 | 0.951701 | 0.989221 | 0.999748 | 22 | 230 | 0 | [[4532, 22, 0], [230, 2019, 1], [0, 0, 3963]] | 0.971288 | good,medium,bad | 0.000146972 | 2223 | 0.000449843 | 1.50162 | 0.976502 | 0.971288 | 0.971288 | 1 | 1 | 0.976502 |
| E4_query_highres_local_art | clean_test | 10767 | 0.973902 | 0.967997 | 0.995389 | 0.884444 | 1 | 0.945754 | 0.989557 | 1 | 21 | 260 | 0 | [[4533, 21, 0], [260, 1990, 0], [0, 0, 3963]] | 0.967997 | good,medium,bad | 0 | 2223 | 0 | 1.45953 | 0.973902 | 0.967997 | 0.967997 | 1 | 1 | 0.973902 |
| E4_query_highres_local_art | clean_test | 2980 | 0.970805 | 0.64451 | 1 | 0.957623 | 0 | 0.914201 | 1 | 0 | 0 | 87 | 0 | [[927, 0, 0], [87, 1966, 0], [0, 0, 0]] | 0.64451 | good,medium,bad | 0 | 788 | 0 | 1.36462 | 0.970805 | 0.966765 | 0.64451 | 0 | nan | nan |
| E0_noquery_nohi_nolocal_noart | clean_test | 10767 | 0.972973 | 0.966756 | 0.996706 | 0.877333 | 1 | 0.942875 | 0.992459 | 0.999748 | 15 | 275 | 0 | [[4539, 15, 0], [275, 1974, 1], [0, 0, 3963]] | 0.966756 | good,medium,bad | 0.000146972 | 2223 | 0.000449843 | 1.44457 | 0.972973 | 0.966756 | 0.966756 | 1 | 1 | 0.972973 |
| E0_noquery_nohi_nolocal_noart | clean_test | 853 | 0.992966 | 0.986125 | 0.992378 | 0.987342 | 1 | 0.998466 | 0.939759 | 1 | 5 | 1 | 0 | [[651, 5, 0], [1, 78, 0], [0, 0, 118]] | 0.986125 | good,medium,bad | 0 | 104 | 0 | 1.05445 | 0.976455 | 0.966465 | 0.564188 | 1 | 1 | 0.995283 |
| E1_query_only | clean_test | 853 | 0.994138 | 0.988247 | 0.995427 | 0.974684 | 1 | 0.996947 | 0.9625 | 1 | 3 | 2 | 0 | [[653, 3, 0], [2, 77, 0], [0, 0, 118]] | 0.988247 | good,medium,bad | 0 | 104 | 0 | 1.0166 | 0.977974 | 0.966302 | 0.563679 | 1 | 1 | 0.995283 |
| E4_query_highres_local_art | clean_test | 853 | 0.991794 | 0.983545 | 0.993902 | 0.962025 | 1 | 0.99542 | 0.95 | 1 | 4 | 3 | 0 | [[652, 4, 0], [3, 76, 0], [0, 0, 118]] | 0.983545 | good,medium,bad | 0 | 104 | 0 | 1.06051 | 0.975921 | 0.964303 | 0.562546 | 1 | 1 | 0.995283 |
| E4_query_highres_local_art | clean_test | 853 | 0.98007 | 0.963243 | 0.974085 | 1 | 1 | 1 | 0.822917 | 1 | 17 | 0 | 0 | [[639, 17, 0], [0, 79, 0], [0, 0, 118]] | 0.963243 | good,medium,bad | 0 | 104 | 0 | 1.3382 | 0.962759 | 0.95992 | 0.562795 | 1 | 1 | 1 |
| E3_query_highres_local | clean_test | 10767 | 0.9674 | 0.95968 | 0.994949 | 0.854222 | 1 | 0.932496 | 0.988175 | 1 | 23 | 328 | 0 | [[4531, 23, 0], [328, 1922, 0], [0, 0, 3963]] | 0.95968 | good,medium,bad | 0 | 2223 | 0 | 1.50088 | 0.9674 | 0.95968 | 0.95968 | 1 | 1 | 0.9674 |
| E1_query_only | clean_test | 10767 | 0.967122 | 0.959301 | 0.995169 | 0.852444 | 1 | 0.931935 | 0.98866 | 0.999748 | 22 | 331 | 0 | [[4532, 22, 0], [331, 1918, 1], [0, 0, 3963]] | 0.959301 | good,medium,bad | 0.000146972 | 2223 | 0.000449843 | 1.41247 | 0.967122 | 0.959301 | 0.959301 | 1 | 1 | 0.967122 |
| E0_noquery_nohi_nolocal_noart | clean_test | 10767 | 0.966286 | 0.958078 | 0.998024 | 0.842667 | 1 | 0.92774 | 0.995276 | 1 | 9 | 354 | 0 | [[4545, 9, 0], [354, 1896, 0], [0, 0, 3963]] | 0.958078 | good,medium,bad | 0 | 2223 | 0 | 1.37915 | 0.966286 | 0.958078 | 0.958078 | 1 | 1 | 0.966286 |
| E3_query_highres_local | clean_test | 2980 | 0.961745 | 0.637984 | 0.998921 | 0.944959 | 0 | 0.8921 | 0.999485 | 0 | 1 | 112 | 0 | [[926, 1, 0], [112, 1940, 1], [0, 0, 0]] | 0.637984 | good,medium,bad | 0.00033557 | 788 | 0.00126904 | 1.40664 | 0.961745 | 0.956975 | 0.637984 | 0 | nan | nan |
| E0_noquery_nohi_nolocal_noart | clean_test | 2980 | 0.960738 | 0.6379 | 1 | 0.94301 | 0 | 0.892204 | 1 | 0 | 0 | 112 | 0 | [[927, 0, 0], [112, 1936, 5], [0, 0, 0]] | 0.6379 | good,medium,bad | 0.00167785 | 788 | 0.00507614 | 1.38373 | 0.960738 | 0.95685 | 0.6379 | 0 | nan | nan |
| E0_noquery_nohi_nolocal_noart | clean_test | 10767 | 0.965264 | 0.95672 | 0.998463 | 0.836889 | 1 | 0.925315 | 0.996296 | 1 | 7 | 367 | 0 | [[4547, 7, 0], [367, 1883, 0], [0, 0, 3963]] | 0.95672 | good,medium,bad | 0 | 2223 | 0 | 1.43638 | 0.965264 | 0.95672 | 0.95672 | 1 | 1 | 0.965264 |
| E1_query_only | clean_test | 10767 | 0.964521 | 0.956078 | 0.992314 | 0.845778 | 1 | 0.928689 | 0.98194 | 1 | 35 | 347 | 0 | [[4519, 35, 0], [347, 1903, 0], [0, 0, 3963]] | 0.956078 | good,medium,bad | 0 | 2223 | 0 | 1.46159 | 0.964521 | 0.956078 | 0.956078 | 1 | 1 | 0.964521 |
| E0_noquery_nohi_nolocal_noart | clean_test | 6126 | 0.95968 | 0.633327 | 0.966121 | 0.942481 | 0 | 0.979527 | 0.912413 | 0 | 151 | 90 | 0 | [[4306, 151, 0], [90, 1573, 6], [0, 0, 0]] | 0.633327 | good,medium,bad | 0.000979432 | 189 | 0.010582 | 1.37664 | 0.95968 | 0.94999 | 0.633327 | 0 | nan | nan |
| E2_query_highres | clean_test | 2980 | 0.955369 | 0.633196 | 1 | 0.935217 | 0 | 0.874528 | 1 | 0 | 0 | 133 | 0 | [[927, 0, 0], [133, 1920, 0], [0, 0, 0]] | 0.633196 | good,medium,bad | 0 | 788 | 0 | 1.30354 | 0.955369 | 0.949794 | 0.633196 | 0 | nan | nan |
| E0_noquery_nohi_nolocal_noart | clean_test | 849 | 0.995289 | 0.66279 | 0.995268 | 1 | 0 | 1 | 0.981651 | 0 | 3 | 0 | 1 | [[631, 3, 0], [0, 214, 0], [0, 1, 0]] | 0.66279 | good,medium,bad | 0 | 222 | 0 | 1.3347 | 0.997287 | 0.948387 | 0.520315 | 1 | 0 | 0.9875 |
| E1_query_only | clean_test | 6126 | 0.959027 | 0.63222 | 0.972403 | 0.923307 | 0 | 0.971531 | 0.926082 | 0 | 123 | 127 | 0 | [[4334, 123, 0], [127, 1541, 1], [0, 0, 0]] | 0.63222 | good,medium,bad | 0.000163239 | 189 | 0.00529101 | 1.32307 | 0.959027 | 0.94833 | 0.63222 | 0 | nan | nan |
| E1_query_only | clean_test | 849 | 0.994111 | 0.661763 | 0.993691 | 1 | 0 | 1 | 0.977169 | 0 | 4 | 0 | 1 | [[630, 4, 0], [0, 214, 0], [0, 1, 0]] | 0.661763 | good,medium,bad | 0 | 222 | 0 | 1.3944 | 0.995682 | 0.94758 | 0.520046 | 1 | 0 | 0.9875 |
| E1_query_only | clean_test | 849 | 0.992933 | 0.660739 | 0.992114 | 1 | 0 | 1 | 0.972727 | 0 | 5 | 0 | 1 | [[629, 5, 0], [0, 214, 0], [0, 1, 0]] | 0.660739 | good,medium,bad | 0 | 222 | 0 | 1.34094 | 0.996429 | 0.947475 | 0.518904 | 1 | 0 | 0.975 |
| E2_query_highres | clean_test | 6126 | 0.958701 | 0.631567 | 0.979134 | 0.904134 | 0 | 0.96506 | 0.941948 | 0 | 93 | 158 | 0 | [[4364, 93, 0], [158, 1509, 2], [0, 0, 0]] | 0.631567 | good,medium,bad | 0.000326477 | 189 | 0.00529101 | 1.24762 | 0.958701 | 0.94735 | 0.631567 | 0 | nan | nan |
| E4_query_highres_local_art | clean_test | 6126 | 0.958211 | 0.631366 | 0.974647 | 0.91432 | 0 | 0.968346 | 0.931056 | 0 | 113 | 142 | 0 | [[4344, 113, 0], [142, 1526, 1], [0, 0, 0]] | 0.631366 | good,medium,bad | 0.000163239 | 189 | 0.00529101 | 1.30007 | 0.958211 | 0.947049 | 0.631366 | 0 | nan | nan |
| E1_query_only | clean_test | 849 | 0.992933 | 0.660739 | 0.992114 | 1 | 0 | 1 | 0.972727 | 0 | 5 | 0 | 1 | [[629, 5, 0], [0, 214, 0], [0, 1, 0]] | 0.660739 | good,medium,bad | 0 | 222 | 0 | 1.38499 | 0.996096 | 0.94676 | 0.518688 | 1 | 0 | 0.979167 |
| E2_query_highres | clean_test | 849 | 0.992933 | 0.660739 | 0.992114 | 1 | 0 | 1 | 0.972727 | 0 | 5 | 0 | 1 | [[629, 5, 0], [0, 214, 0], [0, 1, 0]] | 0.660739 | good,medium,bad | 0 | 222 | 0 | 1.38129 | 0.996096 | 0.94676 | 0.518688 | 1 | 0 | 0.979167 |
| E2_query_highres | clean_test | 2980 | 0.951342 | 0.630577 | 0.998921 | 0.929859 | 0 | 0.867041 | 0.999476 | 0 | 1 | 142 | 0 | [[926, 1, 0], [142, 1909, 2], [0, 0, 0]] | 0.630577 | good,medium,bad | 0.000671141 | 788 | 0.00126904 | 1.39959 | 0.951342 | 0.945866 | 0.630577 | 0 | nan | nan |
| E4_query_highres_local_art | clean_test | 849 | 0.992933 | 0.66072 | 0.993691 | 0.995327 | 0 | 0.998415 | 0.977064 | 0 | 4 | 1 | 1 | [[630, 4, 0], [1, 213, 0], [0, 1, 0]] | 0.66072 | good,medium,bad | 0 | 222 | 0 | 1.37923 | 0.994679 | 0.944774 | 0.5179 | 1 | 0 | 0.983333 |
| E0_noquery_nohi_nolocal_noart | clean_test | 6126 | 0.956415 | 0.629808 | 0.97689 | 0.901738 | 0 | 0.964982 | 0.935945 | 0 | 103 | 158 | 0 | [[4354, 103, 0], [158, 1505, 6], [0, 0, 0]] | 0.629808 | good,medium,bad | 0.000979432 | 189 | 0.010582 | 1.26754 | 0.956415 | 0.944711 | 0.629808 | 0 | nan | nan |
| E2_query_highres | clean_test | 10767 | 0.955791 | 0.944159 | 0.998024 | 0.792444 | 1 | 0.906824 | 0.994978 | 1 | 9 | 467 | 0 | [[4545, 9, 0], [467, 1783, 0], [0, 0, 3963]] | 0.944159 | good,medium,bad | 0 | 2223 | 0 | 1.46145 | 0.955791 | 0.944159 | 0.944159 | 1 | 1 | 0.955791 |
| E0_noquery_nohi_nolocal_noart | clean_test | 6126 | 0.954783 | 0.629218 | 0.962082 | 0.935291 | 0 | 0.976321 | 0.902312 | 0 | 169 | 104 | 0 | [[4288, 169, 0], [104, 1561, 4], [0, 0, 0]] | 0.629218 | good,medium,bad | 0.000652955 | 189 | 0.010582 | 1.32069 | 0.954783 | 0.943827 | 0.629218 | 0 | nan | nan |
| E3_query_highres_local | clean_test | 6126 | 0.95462 | 0.629116 | 0.959614 | 0.941282 | 0 | 0.977824 | 0.897202 | 0 | 180 | 97 | 0 | [[4277, 180, 0], [97, 1571, 1], [0, 0, 0]] | 0.629116 | good,medium,bad | 0.000163239 | 189 | 0.00529101 | 1.43968 | 0.95462 | 0.943673 | 0.629116 | 0 | nan | nan |
| E2_query_highres | clean_test | 6126 | 0.955436 | 0.628794 | 0.976217 | 0.89994 | 0 | 0.96325 | 0.93408 | 0 | 106 | 166 | 0 | [[4351, 106, 0], [166, 1502, 1], [0, 0, 0]] | 0.628794 | good,medium,bad | 0.000163239 | 189 | 0.00529101 | 1.33393 | 0.955436 | 0.943191 | 0.628794 | 0 | nan | nan |
| E3_query_highres_local | clean_test | 6126 | 0.954946 | 0.628699 | 0.972627 | 0.907729 | 0 | 0.966555 | 0.925473 | 0 | 122 | 150 | 0 | [[4335, 122, 0], [150, 1515, 4], [0, 0, 0]] | 0.628699 | good,medium,bad | 0.000652955 | 189 | 0.010582 | 1.3314 | 0.954946 | 0.943049 | 0.628699 | 0 | nan | nan |
| E3_query_highres_local | clean_test | 849 | 0.988221 | 0.656673 | 0.985804 | 1 | 0 | 1 | 0.955357 | 0 | 9 | 0 | 1 | [[625, 9, 0], [0, 214, 0], [0, 1, 0]] | 0.656673 | good,medium,bad | 0 | 222 | 0 | 1.42672 | 0.991363 | 0.942895 | 0.516392 | 1 | 0 | 0.975 |
| E1_query_only | clean_test | 6126 | 0.95364 | 0.628514 | 0.956697 | 0.945476 | 0 | 0.979779 | 0.891022 | 0 | 193 | 88 | 0 | [[4264, 193, 0], [88, 1578, 3], [0, 0, 0]] | 0.628514 | good,medium,bad | 0.000489716 | 189 | 0.010582 | 1.32797 | 0.95364 | 0.942771 | 0.628514 | 0 | nan | nan |
| E4_query_highres_local_art | clean_test | 10767 | 0.954119 | 0.94183 | 0.998902 | 0.782667 | 1 | 0.902938 | 0.997169 | 1 | 5 | 489 | 0 | [[4549, 5, 0], [489, 1761, 0], [0, 0, 3963]] | 0.94183 | good,medium,bad | 0 | 2223 | 0 | 1.42994 | 0.954119 | 0.94183 | 0.94183 | 1 | 1 | 0.954119 |

_Showing 50 of 80 rows._

## Feature Recovery

| candidate | bucket | feature | corr_all | mae | corr_good | corr_medium | corr_bad | corr_min_supported_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0_noquery_nohi_nolocal_noart | clean_test | qrs_visibility | 0.620454 | 1.42228 | -0.245734 | 0.580169 | 0.0356017 | -0.245734 |
| E0_noquery_nohi_nolocal_noart | clean_test | detector_agreement | 0.223644 | 0.139445 | 0.264455 | -0.0609979 | -0.0485787 | -0.0609979 |
| E0_noquery_nohi_nolocal_noart | clean_test | baseline_step | 0.811369 | 0.567426 | 0.822058 | 0.730703 | -0.00235879 | -0.00235879 |
| E0_noquery_nohi_nolocal_noart | clean_test | flatline_ratio | 0.901952 | 2.42198 | 0.833182 | 0.691213 | -0.00167168 | -0.00167168 |
| E0_noquery_nohi_nolocal_noart | clean_test | sqi_basSQI | 0.496356 | 1.93387 | 0.754282 | 0.696323 | 0.00418679 | 0.00418679 |
| E0_noquery_nohi_nolocal_noart | clean_test | non_qrs_diff_p95 | 0.991177 | 0.580267 | 0.59278 | 0.756063 | 0.0298656 | 0.0298656 |
| E0_noquery_nohi_nolocal_noart | clean_test | qrs_band_ratio | -0.826643 | 0.857575 | 0.740861 | 0.621568 | -0.178314 | -0.178314 |
| E0_noquery_nohi_nolocal_noart | clean_test | template_corr | 0.902757 | 0.760328 | 0.21502 | 0.161159 | 0.00338716 | 0.00338716 |
| E0_noquery_nohi_nolocal_noart | clean_test | amplitude_entropy | 0.968301 | 0.474305 | 0.887294 | 0.790533 | 0.445966 | 0.445966 |
| E0_noquery_nohi_nolocal_noart | clean_test | contact_loss_win_ratio | 0.00520439 | 5.20636 | 0.0130121 | nan | nan | 0.0130121 |
| E0_noquery_nohi_nolocal_noart | clean_test | qrs_visibility | 0.672029 | 1.47872 | -0.0729845 | 0.601719 | 0.0857019 | -0.0729845 |
| E0_noquery_nohi_nolocal_noart | clean_test | detector_agreement | 0.0315572 | 0.134525 | 0.320946 | -0.425807 | -0.0439353 | -0.425807 |
| E0_noquery_nohi_nolocal_noart | clean_test | baseline_step | 0.797958 | 0.650148 | 0.853916 | 0.770318 | -0.0217003 | -0.0217003 |
| E0_noquery_nohi_nolocal_noart | clean_test | flatline_ratio | 0.920912 | 2.43773 | 0.858037 | 0.774207 | -0.0243962 | -0.0243962 |
| E0_noquery_nohi_nolocal_noart | clean_test | sqi_basSQI | 0.605152 | 1.92862 | 0.834865 | 0.814752 | -0.00861618 | -0.00861618 |
| E0_noquery_nohi_nolocal_noart | clean_test | non_qrs_diff_p95 | 0.992157 | 0.497628 | 0.777159 | 0.726152 | -0.0382082 | -0.0382082 |
| E0_noquery_nohi_nolocal_noart | clean_test | qrs_band_ratio | -0.81082 | 0.90641 | 0.820551 | 0.667192 | 0.0775606 | 0.0775606 |
| E0_noquery_nohi_nolocal_noart | clean_test | template_corr | 0.907606 | 0.630906 | 0.244238 | 0.271889 | -0.0464772 | -0.0464772 |
| E0_noquery_nohi_nolocal_noart | clean_test | amplitude_entropy | 0.974343 | 0.444987 | 0.922014 | 0.779818 | 0.393817 | 0.393817 |
| E0_noquery_nohi_nolocal_noart | clean_test | contact_loss_win_ratio | 0.00208149 | 5.33607 | 0.0147508 | nan | nan | 0.0147508 |
| E0_noquery_nohi_nolocal_noart | clean_test | qrs_visibility | 0.631044 | 1.48406 | -0.3234 | 0.612424 | 0.0279125 | -0.3234 |
| E0_noquery_nohi_nolocal_noart | clean_test | detector_agreement | 0.25559 | 0.127818 | 0.306585 | 0.0229809 | -0.041493 | -0.041493 |
| E0_noquery_nohi_nolocal_noart | clean_test | baseline_step | 0.875282 | 0.584168 | 0.8663 | 0.792389 | -0.0187635 | -0.0187635 |
| E0_noquery_nohi_nolocal_noart | clean_test | flatline_ratio | 0.913633 | 2.24429 | 0.831188 | 0.696152 | -0.0174097 | -0.0174097 |
| E0_noquery_nohi_nolocal_noart | clean_test | sqi_basSQI | 0.282701 | 1.69131 | 0.793072 | 0.819322 | -0.0122254 | -0.0122254 |
| E0_noquery_nohi_nolocal_noart | clean_test | non_qrs_diff_p95 | 0.991458 | 0.392015 | 0.672456 | 0.747608 | 0.0688978 | 0.0688978 |
| E0_noquery_nohi_nolocal_noart | clean_test | qrs_band_ratio | -0.793674 | 0.862263 | 0.728333 | 0.737982 | -0.139152 | -0.139152 |
| E0_noquery_nohi_nolocal_noart | clean_test | template_corr | 0.904119 | 0.633458 | 0.0754018 | 0.413985 | -0.0362543 | -0.0362543 |
| E0_noquery_nohi_nolocal_noart | clean_test | amplitude_entropy | 0.972955 | 0.461738 | 0.89642 | 0.747359 | 0.270915 | 0.270915 |
| E0_noquery_nohi_nolocal_noart | clean_test | contact_loss_win_ratio | 0.00513852 | 5.31039 | 0.0123668 | nan | nan | 0.0123668 |
| E0_noquery_nohi_nolocal_noart | clean_test | qrs_visibility | 0.649487 | 0.775792 | 0.487773 | 0.590851 | nan | 0.487773 |
| E0_noquery_nohi_nolocal_noart | clean_test | detector_agreement | 0.0146297 | 0.138304 | 0.0993101 | -0.195199 | nan | -0.195199 |
| E0_noquery_nohi_nolocal_noart | clean_test | baseline_step | 0.814342 | 0.334202 | 0.769428 | 0.80241 | nan | 0.769428 |
| E0_noquery_nohi_nolocal_noart | clean_test | flatline_ratio | 0.878719 | 1.67382 | 0.823907 | 0.708327 | nan | 0.708327 |
| E0_noquery_nohi_nolocal_noart | clean_test | sqi_basSQI | 0.763647 | 2.75231 | 0.792927 | 0.802716 | nan | 0.792927 |
| E0_noquery_nohi_nolocal_noart | clean_test | non_qrs_diff_p95 | 0.826595 | 0.157771 | 0.604468 | 0.802172 | nan | 0.604468 |
| E0_noquery_nohi_nolocal_noart | clean_test | qrs_band_ratio | 0.846847 | 0.357532 | 0.848897 | 0.881965 | nan | 0.848897 |
| E0_noquery_nohi_nolocal_noart | clean_test | template_corr | 0.466388 | 0.333888 | 0.509105 | 0.334214 | nan | 0.334214 |
| E0_noquery_nohi_nolocal_noart | clean_test | amplitude_entropy | 0.861446 | 0.272399 | 0.8066 | 0.849827 | nan | 0.8066 |
| E0_noquery_nohi_nolocal_noart | clean_test | contact_loss_win_ratio | 0 | 6.97034 | nan | nan | nan | nan |
| E0_noquery_nohi_nolocal_noart | clean_test | qrs_visibility | 0.480246 | 0.715336 | -0.0896825 | 0.603654 | nan | -0.0896825 |
| E0_noquery_nohi_nolocal_noart | clean_test | detector_agreement | 0.0593757 | 0.162661 | 0.0611177 | -0.151814 | nan | -0.151814 |
| E0_noquery_nohi_nolocal_noart | clean_test | baseline_step | 0.766336 | 0.423484 | 0.744421 | 0.760843 | nan | 0.744421 |
| E0_noquery_nohi_nolocal_noart | clean_test | flatline_ratio | 0.842557 | 1.55258 | 0.786316 | 0.680325 | nan | 0.680325 |
| E0_noquery_nohi_nolocal_noart | clean_test | sqi_basSQI | 0.733692 | 2.42561 | 0.694559 | 0.779037 | nan | 0.694559 |
| E0_noquery_nohi_nolocal_noart | clean_test | non_qrs_diff_p95 | 0.802836 | 0.151475 | 0.446891 | 0.787513 | nan | 0.446891 |
| E0_noquery_nohi_nolocal_noart | clean_test | qrs_band_ratio | 0.776596 | 0.485185 | 0.846343 | 0.808977 | nan | 0.808977 |
| E0_noquery_nohi_nolocal_noart | clean_test | template_corr | 0.39114 | 0.135576 | 0.399412 | 0.360777 | nan | 0.360777 |
| E0_noquery_nohi_nolocal_noart | clean_test | amplitude_entropy | 0.840718 | 0.259137 | 0.762481 | 0.863855 | nan | 0.762481 |
| E0_noquery_nohi_nolocal_noart | clean_test | contact_loss_win_ratio | 0 | 6.36432 | nan | nan | nan | nan |

_Showing 50 of 80 rows._

## Record Metrics Preview

| candidate | record_id | rows | good_rows | medium_rows | bad_rows | full_macro_f1 | supported_macro_f1 | acc | good_recall | medium_recall | bad_recall | artifact_positive_nonbad | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0_noquery_nohi_nolocal_noart | 111001 | 2980 | 927 | 2053 | 0 | 0.613375 | 0.920063 | 0.927517 | 1 | 0.894788 | 0 | 1133 | 0 |
| E0_noquery_nohi_nolocal_noart | 123001 | 92 | 90 | 2 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 7 | 0 |
| E0_noquery_nohi_nolocal_noart | 125001 | 7 | 7 | 0 | 0 | 0.307692 | 0.923077 | 0.857143 | 0.857143 | 0 | 0 | 1 | 0 |
| E0_noquery_nohi_nolocal_noart | 105001 | 10767 | 4554 | 2250 | 3963 | 0.95672 | 0.95672 | 0.965264 | 0.998463 | 0.836889 | 1 | 2223 | 0 |
| E0_noquery_nohi_nolocal_noart | 111001 | 2980 | 927 | 2053 | 0 | 0.623617 | 0.935426 | 0.941611 | 1 | 0.915246 | 0 | 1133 | 0.00176523 |
| E0_noquery_nohi_nolocal_noart | 123001 | 92 | 90 | 2 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 7 | 0 |
| E0_noquery_nohi_nolocal_noart | 125001 | 7 | 7 | 0 | 0 | 0.307692 | 0.923077 | 0.857143 | 0.857143 | 0 | 0 | 1 | 0 |
| E0_noquery_nohi_nolocal_noart | 105001 | 10767 | 4554 | 2250 | 3963 | 0.966756 | 0.966756 | 0.972973 | 0.996706 | 0.877333 | 1 | 2223 | 0.000449843 |
| E0_noquery_nohi_nolocal_noart | 111001 | 2980 | 927 | 2053 | 0 | 0.618471 | 0.927707 | 0.934564 | 1 | 0.905017 | 0 | 1133 | 0.000882613 |
| E0_noquery_nohi_nolocal_noart | 123001 | 92 | 90 | 2 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 7 | 0 |
| E0_noquery_nohi_nolocal_noart | 125001 | 7 | 7 | 0 | 0 | 0.307692 | 0.923077 | 0.857143 | 0.857143 | 0 | 0 | 1 | 0 |
| E0_noquery_nohi_nolocal_noart | 105001 | 10767 | 4554 | 2250 | 3963 | 0.958078 | 0.958078 | 0.966286 | 0.998024 | 0.842667 | 1 | 2223 | 0 |
| E0_noquery_nohi_nolocal_noart | 104001 | 42 | 0 | 42 | 0 | 0.333333 | 1 | 1 | 0 | 1 | 0 | 10 | 0 |
| E0_noquery_nohi_nolocal_noart | 113001 | 154 | 125 | 29 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 25 | 0 |
| E0_noquery_nohi_nolocal_noart | 122001 | 212 | 70 | 24 | 118 | 0.983818 | 0.983818 | 0.990566 | 1 | 1 | 0.983051 | 1 | 0 |
| E0_noquery_nohi_nolocal_noart | 100001 | 6126 | 4457 | 1669 | 0 | 0.633327 | 0.94999 | 0.95968 | 0.966121 | 0.942481 | 0 | 189 | 0.010582 |
| E0_noquery_nohi_nolocal_noart | 104001 | 42 | 0 | 42 | 0 | 0.333333 | 1 | 1 | 0 | 1 | 0 | 10 | 0 |
| E0_noquery_nohi_nolocal_noart | 113001 | 154 | 125 | 29 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 25 | 0 |
| E0_noquery_nohi_nolocal_noart | 122001 | 212 | 70 | 24 | 118 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| E0_noquery_nohi_nolocal_noart | 100001 | 6126 | 4457 | 1669 | 0 | 0.629808 | 0.944711 | 0.956415 | 0.97689 | 0.901738 | 0 | 189 | 0.010582 |
| E0_noquery_nohi_nolocal_noart | 104001 | 42 | 0 | 42 | 0 | 0.333333 | 1 | 1 | 0 | 1 | 0 | 10 | 0 |
| E0_noquery_nohi_nolocal_noart | 113001 | 154 | 125 | 29 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 25 | 0 |
| E0_noquery_nohi_nolocal_noart | 122001 | 212 | 70 | 24 | 118 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| E0_noquery_nohi_nolocal_noart | 100001 | 6126 | 4457 | 1669 | 0 | 0.629218 | 0.943827 | 0.954783 | 0.962082 | 0.935291 | 0 | 189 | 0.010582 |
| E0_noquery_nohi_nolocal_noart | 100002 | 79 | 56 | 23 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 12 | 0 |
| E0_noquery_nohi_nolocal_noart | 124001 | 79 | 0 | 79 | 0 | 0.333333 | 1 | 1 | 0 | 1 | 0 | 18 | 0 |
| E0_noquery_nohi_nolocal_noart | 126001 | 105 | 105 | 0 | 0 | 0.333333 | 1 | 1 | 1 | 0 | 0 | 0 | nan |
| E0_noquery_nohi_nolocal_noart | 111001 | 2980 | 927 | 2053 | 0 | 0.618386 | 0.927579 | 0.933221 | 0.998921 | 0.903556 | 0 | 788 | 0.00507614 |
| E0_noquery_nohi_nolocal_noart | 100002 | 79 | 56 | 23 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 12 | 0 |
| E0_noquery_nohi_nolocal_noart | 124001 | 79 | 0 | 79 | 0 | 0.333333 | 1 | 1 | 0 | 1 | 0 | 18 | 0 |
| E0_noquery_nohi_nolocal_noart | 126001 | 105 | 105 | 0 | 0 | 0.333333 | 1 | 1 | 1 | 0 | 0 | 0 | nan |
| E0_noquery_nohi_nolocal_noart | 111001 | 2980 | 927 | 2053 | 0 | 0.576442 | 0.864664 | 0.872819 | 1 | 0.815392 | 0 | 788 | 0 |
| E0_noquery_nohi_nolocal_noart | 100002 | 79 | 56 | 23 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 12 | 0 |
| E0_noquery_nohi_nolocal_noart | 124001 | 79 | 0 | 79 | 0 | 0.333333 | 1 | 1 | 0 | 1 | 0 | 18 | 0 |
| E0_noquery_nohi_nolocal_noart | 126001 | 105 | 105 | 0 | 0 | 0.333333 | 1 | 1 | 1 | 0 | 0 | 0 | nan |
| E0_noquery_nohi_nolocal_noart | 111001 | 2980 | 927 | 2053 | 0 | 0.6379 | 0.95685 | 0.960738 | 1 | 0.94301 | 0 | 788 | 0.00507614 |
| E0_noquery_nohi_nolocal_noart | 118001 | 145 | 116 | 29 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 1 | 0 |
| E0_noquery_nohi_nolocal_noart | 126001 | 105 | 105 | 0 | 0 | 0.333333 | 1 | 1 | 1 | 0 | 0 | 0 | nan |
| E0_noquery_nohi_nolocal_noart | 103001 | 107 | 95 | 12 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 86 | 0 |
| E0_noquery_nohi_nolocal_noart | 104001 | 42 | 0 | 42 | 0 | 0.333333 | 1 | 1 | 0 | 1 | 0 | 10 | 0 |
| E0_noquery_nohi_nolocal_noart | 113001 | 154 | 125 | 29 | 0 | 0.659678 | 0.989517 | 0.993506 | 0.992 | 1 | 0 | 29 | 0 |
| E0_noquery_nohi_nolocal_noart | 114001 | 240 | 209 | 30 | 1 | 0.649191 | 0.649191 | 0.9875 | 0.990431 | 1 | 0 | 18 | 0 |
| E0_noquery_nohi_nolocal_noart | 115001 | 138 | 116 | 22 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 2 | 0 |
| E0_noquery_nohi_nolocal_noart | 121001 | 89 | 89 | 0 | 0 | 0.333333 | 1 | 1 | 1 | 0 | 0 | 63 | 0 |
| E0_noquery_nohi_nolocal_noart | 124001 | 79 | 0 | 79 | 0 | 0.333333 | 1 | 1 | 0 | 1 | 0 | 14 | 0 |
| E0_noquery_nohi_nolocal_noart | 118001 | 145 | 116 | 29 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 1 | 0 |
| E0_noquery_nohi_nolocal_noart | 126001 | 105 | 105 | 0 | 0 | 0.128205 | 0.384615 | 0.238095 | 0.238095 | 0 | 0 | 0 | nan |
| E0_noquery_nohi_nolocal_noart | 103001 | 107 | 95 | 12 | 0 | 0.666667 | 1 | 1 | 1 | 1 | 0 | 86 | 0 |
| E0_noquery_nohi_nolocal_noart | 104001 | 42 | 0 | 42 | 0 | 0.333333 | 1 | 1 | 0 | 1 | 0 | 10 | 0 |
| E0_noquery_nohi_nolocal_noart | 113001 | 154 | 125 | 29 | 0 | 0.659678 | 0.989517 | 0.993506 | 0.992 | 1 | 0 | 29 | 0 |

_Showing 50 of 80 rows._

## Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_feature_recovery.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_record_metrics.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_sqi_conformer`