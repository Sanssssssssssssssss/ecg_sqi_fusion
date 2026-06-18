# BUT Split Roll Audit

External-only diagnostic. Strict candidates split by `subject_id`; the window-random split is an explicitly leaky capacity upper bound.

## Selected Strict Splits

| balanced_best_candidate_seed | hard_test_candidate_seed | balanced_best_score | hard_test_score | hard_test_hard_score | window_random_seed | strict_split_unit | diagnostic_window_random_is_not_external_claim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4403 | 4819 | 5.70359 | 7.57436 | 0.721813 | 20260618 | subject_id | True |

## Best Candidate Rows

| candidate_seed | score | hard_score | train_subjects | val_subjects | test_subjects | train_n | train_prop | train_subject_count | train_record_count | train_max_record_share | train_bad | train_good | train_medium | train_clean_core | train_good_medium_overlap | train_medium_bad_overlap | train_near_bad_boundary | train_outlier_low_confidence | val_n | val_prop | val_subject_count | val_record_count | val_max_record_share | val_bad | val_good | val_medium | val_clean_core | val_good_medium_overlap | val_medium_bad_overlap | val_near_bad_boundary | val_outlier_low_confidence | test_n | test_prop | test_subject_count | test_record_count | test_max_record_share | test_bad | test_good | test_medium | test_clean_core | test_good_medium_overlap | test_medium_bad_overlap | test_near_bad_boundary | test_outlier_low_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 184 | 4.85183 | 0.441466 | 103,105,111,115,118,121,125,126 | 113,122,123,124 | 100,104,114 | 22875 | 0.694107 | 8 | 8 | 0.580721 | 5026 | 10237 | 7612 | 3527 | 8493 | 9 | 0 | 6882 | 988 | 0.0299794 | 4 | 4 | 0.281377 | 154 | 541 | 293 | 139 | 437 | 0 | 119 | 293 | 9093 | 0.275913 | 3 | 3 | 0.919828 | 105 | 6265 | 2723 | 1568 | 5440 | 0 | 1 | 2084 |
| 899 | 4.88674 | 0.422627 | 103,105,111,114,118,123,125,126 | 100,104,113 | 115,121,122,124 | 23084 | 0.700449 | 8 | 8 | 0.575464 | 5109 | 10361 | 7614 | 3494 | 8594 | 9 | 1 | 7022 | 9008 | 0.273334 | 3 | 3 | 0.928508 | 26 | 6235 | 2747 | 1595 | 5357 | 0 | 0 | 2056 | 864 | 0.0262168 | 4 | 4 | 0.31713 | 150 | 447 | 267 | 145 | 419 | 0 | 119 | 181 |
| 4692 | 4.94444 | 0.442725 | 105,111,115,118,121,125,126 | 113,122,123,124 | 100,103,104,114 | 22305 | 0.676812 | 7 | 7 | 0.595562 | 5026 | 9721 | 7558 | 3395 | 8225 | 9 | 0 | 6712 | 988 | 0.0299794 | 4 | 4 | 0.281377 | 154 | 541 | 293 | 139 | 437 | 0 | 119 | 293 | 9663 | 0.293209 | 4 | 4 | 0.86557 | 105 | 6781 | 2777 | 1700 | 5708 | 0 | 1 | 2254 |
| 4018 | 4.96876 | 0.451294 | 103,104,105,111,118,125,126 | 100,113,114,121 | 115,122,123,124 | 22719 | 0.689374 | 7 | 7 | 0.584709 | 5026 | 10072 | 7621 | 3493 | 8275 | 9 | 0 | 6978 | 9348 | 0.283651 | 4 | 4 | 0.894737 | 109 | 6495 | 2744 | 1596 | 5693 | 0 | 1 | 2058 | 889 | 0.0269754 | 4 | 4 | 0.308211 | 150 | 476 | 263 | 145 | 402 | 0 | 119 | 223 |
| 5613 | 5.01275 | 0.451294 | 103,104,105,111,113,114,118,121,125 | 100,126 | 115,122,123,124 | 23316 | 0.707489 | 9 | 9 | 0.569738 | 5113 | 10443 | 7760 | 3552 | 8678 | 9 | 1 | 7112 | 8751 | 0.265536 | 2 | 2 | 0.955776 | 22 | 6124 | 2605 | 1537 | 5290 | 0 | 0 | 1924 | 889 | 0.0269754 | 4 | 4 | 0.308211 | 150 | 476 | 263 | 145 | 402 | 0 | 119 | 223 |
| 17017 | 5.0378 | 0.437106 | 103,105,111,114,115,123,125,126 | 118,121,122,124 | 100,104,113 | 23071 | 0.700055 | 8 | 8 | 0.575788 | 5109 | 10364 | 7598 | 3516 | 8555 | 9 | 1 | 7026 | 877 | 0.0266112 | 4 | 4 | 0.312429 | 150 | 444 | 283 | 123 | 458 | 0 | 119 | 177 | 9008 | 0.273334 | 3 | 3 | 0.928508 | 26 | 6235 | 2747 | 1595 | 5357 | 0 | 0 | 2056 |
| 17191 | 5.06929 | 0.507374 | 100,103,105,111,114,118,121,123,125 | 115,124,126 | 104,113,122 | 31567 | 0.957853 | 9 | 9 | 0.420819 | 5131 | 16204 | 10232 | 5045 | 13754 | 9 | 1 | 8794 | 677 | 0.0205425 | 3 | 3 | 0.404727 | 31 | 428 | 218 | 102 | 346 | 0 | 0 | 229 | 712 | 0.0216046 | 3 | 3 | 0.390449 | 123 | 411 | 178 | 87 | 270 | 0 | 119 | 236 |
| 13873 | 5.07238 | 0.449963 | 103,105,111,113,121,122,126 | 114,115,124 | 100,104,118,123,125 | 22785 | 0.691376 | 7 | 7 | 0.583015 | 5149 | 10019 | 7617 | 3530 | 8431 | 9 | 119 | 6732 | 816 | 0.0247603 | 3 | 3 | 0.444853 | 114 | 433 | 269 | 111 | 465 | 0 | 1 | 239 | 9355 | 0.283863 | 5 | 5 | 0.894067 | 22 | 6591 | 2742 | 1593 | 5474 | 0 | 0 | 2288 |
| 19899 | 5.07238 | 0.449963 | 103,105,111,113,121,122,126 | 114,115,124 | 100,104,118,123,125 | 22785 | 0.691376 | 7 | 7 | 0.583015 | 5149 | 10019 | 7617 | 3530 | 8431 | 9 | 119 | 6732 | 816 | 0.0247603 | 3 | 3 | 0.444853 | 114 | 433 | 269 | 111 | 465 | 0 | 1 | 239 | 9355 | 0.283863 | 5 | 5 | 0.894067 | 22 | 6591 | 2742 | 1593 | 5474 | 0 | 0 | 2288 |
| 12531 | 5.07551 | 0.433453 | 103,105,111,115,118,123,125,126 | 104,113,121,122,124 | 100,114 | 22900 | 0.694866 | 8 | 8 | 0.580087 | 5026 | 10266 | 7608 | 3527 | 8476 | 9 | 0 | 6924 | 1166 | 0.0353805 | 5 | 5 | 0.238422 | 154 | 648 | 364 | 161 | 485 | 0 | 119 | 401 | 8890 | 0.269754 | 2 | 2 | 0.940832 | 105 | 6129 | 2656 | 1546 | 5409 | 0 | 1 | 1934 |
| 6529 | 5.07676 | 0.440324 | 103,104,105,111,123,126 | 100,114,118,125 | 113,115,121,122,124 | 22504 | 0.68285 | 6 | 6 | 0.590295 | 5026 | 9909 | 7569 | 3487 | 8206 | 9 | 0 | 6838 | 9310 | 0.282498 | 4 | 4 | 0.898389 | 105 | 6488 | 2717 | 1566 | 5583 | 0 | 1 | 2160 | 1142 | 0.0346523 | 5 | 5 | 0.243433 | 154 | 646 | 342 | 181 | 581 | 0 | 119 | 261 |
| 13040 | 5.09024 | 0.442754 | 103,104,105,111,114,115,123,126 | 118,121,122,124 | 100,113,125 | 23046 | 0.699296 | 8 | 8 | 0.576412 | 5109 | 10272 | 7665 | 3538 | 8578 | 9 | 1 | 6956 | 877 | 0.0266112 | 4 | 4 | 0.312429 | 150 | 444 | 283 | 123 | 458 | 0 | 119 | 177 | 9033 | 0.274093 | 3 | 3 | 0.925938 | 26 | 6327 | 2680 | 1573 | 5334 | 0 | 0 | 2126 |

## Counts Snapshot

| scheme | split | class_name | n |
| --- | --- | --- | --- |
| balanced_best | test | bad | 105 |
| balanced_best | test | good | 6296 |
| balanced_best | test | medium | 2669 |
| balanced_best | train | bad | 5030 |
| balanced_best | train | good | 9619 |
| balanced_best | train | medium | 7575 |
| balanced_best | val | bad | 150 |
| balanced_best | val | good | 1128 |
| balanced_best | val | medium | 384 |
| current | test | bad | 411 |
| current | test | good | 3640 |
| current | test | medium | 4426 |
| current | train | bad | 4791 |
| current | train | good | 12434 |
| current | train | medium | 6097 |
| current | val | bad | 83 |
| current | val | good | 969 |
| current | val | medium | 105 |
| hard_test | test | bad | 292 |
| hard_test | test | good | 3700 |
| hard_test | test | medium | 4457 |
| hard_test | train | bad | 4940 |
| hard_test | train | good | 7373 |
| hard_test | train | medium | 3393 |
| hard_test | val | bad | 53 |
| hard_test | val | good | 5970 |
| hard_test | val | medium | 2778 |
| window_random_diagnostic | test | bad | 1057 |
| window_random_diagnostic | test | good | 3408 |
| window_random_diagnostic | test | medium | 2127 |
| window_random_diagnostic | train | bad | 3171 |
| window_random_diagnostic | train | good | 10226 |
| window_random_diagnostic | train | medium | 6376 |
| window_random_diagnostic | val | bad | 1057 |
| window_random_diagnostic | val | good | 3409 |
| window_random_diagnostic | val | medium | 2125 |

## Contract

- `balanced_best_split` and `hard_test_split` are subject-level and suitable for BUT-only capacity diagnostics.
- `window_random_diagnostic_split` is only an upper-bound sanity check and must not be used as an external-test claim.
- PTB->BUT selection still cannot use any BUT split.
