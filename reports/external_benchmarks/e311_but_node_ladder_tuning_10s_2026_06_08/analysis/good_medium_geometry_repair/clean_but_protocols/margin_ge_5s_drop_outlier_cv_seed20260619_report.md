# Clean BUT Stratified CV Folds

These folds reuse the already-cleaned fixed-10s windows and rewrite only split labels.
They are window-level folds stratified by record, class, and original_region, not record-heldout external tests.
The goal is stable clean-BUT model selection after dropping low-confidence/outlier windows.

## Global Fold Assignment

| fold_id | bad:near_bad_boundary | bad:right_bad_island | good:clean_core | good:good_medium_overlap | medium:clean_core | medium:good_medium_overlap | medium:medium_bad_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 25 | 793 | 638 | 1620 | 348 | 914 | 2 |
| 1 | 24 | 793 | 635 | 1616 | 345 | 911 | 2 |
| 2 | 24 | 793 | 633 | 1611 | 343 | 910 | 2 |
| 3 | 23 | 792 | 630 | 1609 | 340 | 906 | 2 |
| 4 | 23 | 792 | 628 | 1608 | 337 | 902 | 1 |

## Policies

### margin_ge_5s_drop_outlier_cv_seed20260619_fold0

| split | n | records | good | medium | bad | region_good_medium_overlap | region_clean_core | region_right_bad_island | region_near_bad_boundary | region_medium_bad_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 4340 | 18 | 2258 | 1264 | 818 | 2534 | 986 | 793 | 25 | 2 |
| train | 12909 | 18 | 6719 | 3743 | 2447 | 7546 | 2911 | 2377 | 70 | 5 |
| val | 4326 | 18 | 2251 | 1258 | 817 | 2527 | 980 | 793 | 24 | 2 |

### margin_ge_5s_drop_outlier_cv_seed20260619_fold1

| split | n | records | good | medium | bad | region_good_medium_overlap | region_clean_core | region_right_bad_island | region_near_bad_boundary | region_medium_bad_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 4326 | 18 | 2251 | 1258 | 817 | 2527 | 980 | 793 | 24 | 2 |
| train | 12933 | 18 | 6733 | 3752 | 2448 | 7559 | 2921 | 2377 | 71 | 5 |
| val | 4316 | 18 | 2244 | 1255 | 817 | 2521 | 976 | 793 | 24 | 2 |

### margin_ge_5s_drop_outlier_cv_seed20260619_fold2

| split | n | records | good | medium | bad | region_good_medium_overlap | region_clean_core | region_right_bad_island | region_near_bad_boundary | region_medium_bad_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 4316 | 18 | 2244 | 1255 | 817 | 2521 | 976 | 793 | 24 | 2 |
| train | 12957 | 18 | 6745 | 3762 | 2450 | 7571 | 2931 | 2378 | 72 | 5 |
| val | 4302 | 18 | 2239 | 1248 | 815 | 2515 | 970 | 792 | 23 | 2 |

### margin_ge_5s_drop_outlier_cv_seed20260619_fold3

| split | n | records | good | medium | bad | region_good_medium_overlap | region_clean_core | region_right_bad_island | region_near_bad_boundary | region_medium_bad_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 4302 | 18 | 2239 | 1248 | 815 | 2515 | 970 | 792 | 23 | 2 |
| train | 12982 | 18 | 6753 | 3777 | 2452 | 7582 | 2942 | 2379 | 73 | 6 |
| val | 4291 | 18 | 2236 | 1240 | 815 | 2510 | 965 | 792 | 23 | 1 |

### margin_ge_5s_drop_outlier_cv_seed20260619_fold4

| split | n | records | good | medium | bad | region_good_medium_overlap | region_clean_core | region_right_bad_island | region_near_bad_boundary | region_medium_bad_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 4291 | 18 | 2236 | 1240 | 815 | 2510 | 965 | 792 | 23 | 1 |
| train | 12944 | 18 | 6734 | 3761 | 2449 | 7563 | 2926 | 2378 | 71 | 6 |
| val | 4340 | 18 | 2258 | 1264 | 818 | 2534 | 986 | 793 | 25 | 2 |
