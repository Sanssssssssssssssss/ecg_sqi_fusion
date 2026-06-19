# Clean BUT Stratified CV Folds

These folds reuse the already-cleaned fixed-10s windows and rewrite only split labels.
They are window-level folds stratified by record, class, and original_region, not record-heldout external tests.
The goal is stable clean-BUT model selection after dropping low-confidence/outlier windows.

## Global Fold Assignment

| fold_id | bad:near_bad_boundary | bad:outlier_low_confidence | bad:right_bad_island | good:clean_core | good:good_medium_overlap | good:outlier_low_confidence | medium:clean_core | medium:good_medium_overlap | medium:medium_bad_overlap | medium:outlier_low_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 25 | 217 | 793 | 638 | 1620 | 770 | 348 | 914 | 2 | 596 |
| 1 | 24 | 216 | 793 | 635 | 1616 | 768 | 345 | 911 | 2 | 590 |
| 2 | 24 | 216 | 793 | 633 | 1611 | 762 | 343 | 910 | 2 | 589 |
| 3 | 23 | 213 | 792 | 630 | 1609 | 760 | 340 | 906 | 2 | 587 |
| 4 | 23 | 212 | 792 | 628 | 1608 | 754 | 337 | 902 | 1 | 585 |

## Policies

### margin_ge_5s_keep_outlier_cv_seed20260619_fold0

| split | n | records | good | medium | bad | region_good_medium_overlap | region_outlier_low_confidence | region_clean_core | region_right_bad_island | region_near_bad_boundary | region_medium_bad_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 5923 | 18 | 3028 | 1860 | 1035 | 2534 | 1583 | 986 | 793 | 25 | 2 |
| train | 17587 | 18 | 8995 | 5504 | 3088 | 7546 | 4678 | 2911 | 2377 | 70 | 5 |
| val | 5900 | 18 | 3019 | 1848 | 1033 | 2527 | 1574 | 980 | 793 | 24 | 2 |

### margin_ge_5s_keep_outlier_cv_seed20260619_fold1

| split | n | records | good | medium | bad | region_good_medium_overlap | region_outlier_low_confidence | region_clean_core | region_right_bad_island | region_near_bad_boundary | region_medium_bad_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 5900 | 18 | 3019 | 1848 | 1033 | 2527 | 1574 | 980 | 793 | 24 | 2 |
| train | 17627 | 18 | 9017 | 5520 | 3090 | 7559 | 4694 | 2921 | 2377 | 71 | 5 |
| val | 5883 | 18 | 3006 | 1844 | 1033 | 2521 | 1567 | 976 | 793 | 24 | 2 |

### margin_ge_5s_keep_outlier_cv_seed20260619_fold2

| split | n | records | good | medium | bad | region_good_medium_overlap | region_outlier_low_confidence | region_clean_core | region_right_bad_island | region_near_bad_boundary | region_medium_bad_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 5883 | 18 | 3006 | 1844 | 1033 | 2521 | 1567 | 976 | 793 | 24 | 2 |
| train | 17665 | 18 | 9037 | 5533 | 3095 | 7571 | 4708 | 2931 | 2378 | 72 | 5 |
| val | 5862 | 18 | 2999 | 1835 | 1028 | 2515 | 1560 | 970 | 792 | 23 | 2 |

### margin_ge_5s_keep_outlier_cv_seed20260619_fold3

| split | n | records | good | medium | bad | region_good_medium_overlap | region_outlier_low_confidence | region_clean_core | region_right_bad_island | region_near_bad_boundary | region_medium_bad_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 5862 | 18 | 2999 | 1835 | 1028 | 2515 | 1560 | 970 | 792 | 23 | 2 |
| train | 17706 | 18 | 9053 | 5552 | 3101 | 7582 | 4724 | 2942 | 2379 | 73 | 6 |
| val | 5842 | 18 | 2990 | 1825 | 1027 | 2510 | 1551 | 965 | 792 | 23 | 1 |

### margin_ge_5s_keep_outlier_cv_seed20260619_fold4

| split | n | records | good | medium | bad | region_good_medium_overlap | region_outlier_low_confidence | region_clean_core | region_right_bad_island | region_near_bad_boundary | region_medium_bad_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 5842 | 18 | 2990 | 1825 | 1027 | 2510 | 1551 | 965 | 792 | 23 | 1 |
| train | 17645 | 18 | 9024 | 5527 | 3094 | 7563 | 4701 | 2926 | 2378 | 71 | 6 |
| val | 5923 | 18 | 3028 | 1860 | 1035 | 2534 | 1583 | 986 | 793 | 25 | 2 |
