# Waveform Ensemble Probe V2

Report-only diagnostic.  All members are PTB-trained waveform-only Transformer candidates.
The ensembles below are fixed probability averages; original BUT is not used for training.

## Original Test 10s+

| name | bucket | n | acc | good_recall | medium_recall | bad_recall | good_to_medium | medium_to_good | bad_to_good | bad_to_medium |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| featurefirst_top20_hardrec_a050 | original_test_all_10s+ | 8477 | 0.844520 | 0.848352 | 0.894261 | 0.274939 | 550 | 451 | 169 | 129 |
| equal_hardrec_goodguard_badlite | original_test_all_10s+ | 8477 | 0.843223 | 0.873352 | 0.868278 | 0.306569 | 453 | 563 | 190 | 95 |
| equal_hardrec_badlite | original_test_all_10s+ | 8477 | 0.838032 | 0.889835 | 0.845459 | 0.299270 | 398 | 664 | 200 | 88 |
| equal_hardrec_qrslite | original_test_all_10s+ | 8477 | 0.837560 | 0.871429 | 0.863986 | 0.253041 | 468 | 589 | 194 | 113 |
| featurefirst_top20_shift_stress_a050 | original_test_all_10s+ | 8477 | 0.837325 | 0.834066 | 0.885450 | 0.347932 | 582 | 411 | 188 | 80 |
| equal_top2_old_featurefirst | original_test_all_10s+ | 8477 | 0.834257 | 0.853022 | 0.866697 | 0.318735 | 525 | 547 | 197 | 83 |
| equal_hardrec_badlite_qrsfocus | original_test_all_10s+ | 8477 | 0.832252 | 0.887363 | 0.835291 | 0.311436 | 406 | 686 | 194 | 89 |
| equal_hardrec_all_badlite | original_test_all_10s+ | 8477 | 0.831308 | 0.888736 | 0.833484 | 0.299270 | 402 | 717 | 201 | 87 |
| equal_old_featurefirst_sparsebad | original_test_all_10s+ | 8477 | 0.830836 | 0.872253 | 0.843425 | 0.328467 | 446 | 618 | 193 | 83 |
| equal_featurefirst_sparsebad | original_test_all_10s+ | 8477 | 0.828595 | 0.876099 | 0.835291 | 0.335766 | 423 | 616 | 192 | 81 |
| featurefirst_top20_hardrec_goodguard_badlite_a050 | original_test_all_10s+ | 8477 | 0.826590 | 0.879945 | 0.831676 | 0.299270 | 425 | 684 | 203 | 85 |
| equal_top4_old_p20_burstdrop_sparse | original_test_all_10s+ | 8477 | 0.823758 | 0.875824 | 0.828287 | 0.313869 | 442 | 717 | 194 | 88 |
| featurefirst_top20_hardrec_qrslite_a050 | original_test_all_10s+ | 8477 | 0.823758 | 0.885440 | 0.827610 | 0.236010 | 417 | 758 | 215 | 99 |
| predtop20_sqiquery_subject111_shift_stress_pretrain | original_test_all_10s+ | 8477 | 0.823051 | 0.851923 | 0.847266 | 0.306569 | 531 | 635 | 205 | 80 |
| predtop20_sqiquery_subject111_impulsebad_dual_p20 | original_test_all_10s+ | 8477 | 0.822225 | 0.857967 | 0.844103 | 0.270073 | 512 | 667 | 190 | 110 |
| equal_top3_old_p20_qrsstress | original_test_all_10s+ | 8477 | 0.821517 | 0.869505 | 0.832128 | 0.282238 | 466 | 715 | 196 | 99 |
| featurefirst_top20_shift_stress_a050_seed18 | original_test_all_10s+ | 8477 | 0.817978 | 0.905495 | 0.792363 | 0.318735 | 336 | 863 | 217 | 63 |
| predtop20_sqiquery_subject111_burstdrop_dual_p26 | original_test_all_10s+ | 8477 | 0.816916 | 0.885440 | 0.810664 | 0.277372 | 410 | 806 | 200 | 97 |
| equal_top2_old_qrsstress | original_test_all_10s+ | 8477 | 0.816327 | 0.875549 | 0.815861 | 0.296837 | 445 | 779 | 207 | 82 |
| equal_all_structural | original_test_all_10s+ | 8477 | 0.815855 | 0.892308 | 0.800045 | 0.309002 | 385 | 860 | 212 | 72 |
| predtop20_sqiquery_subject111_sparseevent_v5_badonly | original_test_all_10s+ | 8477 | 0.815501 | 0.881044 | 0.806146 | 0.335766 | 395 | 728 | 194 | 79 |
| equal_old_plus_featurefirst_seeds | original_test_all_10s+ | 8477 | 0.812670 | 0.898077 | 0.789200 | 0.309002 | 365 | 904 | 221 | 63 |
| featurefirst_top20_hardrec_qrsfocus_a050 | original_test_all_10s+ | 8477 | 0.812080 | 0.874451 | 0.812246 | 0.257908 | 443 | 737 | 188 | 117 |
| equal_featurefirst_a050_seeds | original_test_all_10s+ | 8477 | 0.807361 | 0.904396 | 0.774062 | 0.306569 | 341 | 972 | 225 | 60 |
| predtop20_sqiquery_qrsstressv3_stress_pretrain | original_test_all_10s+ | 8477 | 0.804530 | 0.895055 | 0.781066 | 0.255474 | 377 | 928 | 195 | 111 |
| equal_old_plus_structural | original_test_all_10s+ | 8477 | 0.802171 | 0.883516 | 0.783326 | 0.284672 | 419 | 935 | 219 | 75 |
| featurefirst_top20_hardrec_badlite_a050 | original_test_all_10s+ | 8477 | 0.798042 | 0.911538 | 0.748531 | 0.326034 | 312 | 966 | 227 | 50 |
| featurefirst_top20_shift_stress_a050_seed20 | original_test_all_10s+ | 8477 | 0.795093 | 0.900275 | 0.756213 | 0.282238 | 355 | 1051 | 234 | 61 |
| featurefirst_top20_shift_stress_a050_seed19 | original_test_all_10s+ | 8477 | 0.793087 | 0.905769 | 0.747854 | 0.282238 | 342 | 1104 | 234 | 61 |
| featurefirst_top20_shift_stress_a050_seed21 | original_test_all_10s+ | 8477 | 0.790846 | 0.906044 | 0.742205 | 0.294404 | 342 | 1131 | 223 | 67 |
| predtop20_sqiquery_qrsstressv3_pretrain | original_test_all_10s+ | 8477 | 0.790020 | 0.882418 | 0.775644 | 0.126521 | 428 | 989 | 224 | 135 |
| predtop20_sqiquery_primtree_stress_teacher_pretrain | original_test_all_10s+ | 8477 | 0.777398 | 0.903571 | 0.723452 | 0.240876 | 351 | 1224 | 245 | 67 |
| predtop20_sqiquery_thresholdtree_badguard_pretrain | original_test_all_10s+ | 8477 | 0.766545 | 0.887363 | 0.711026 | 0.294404 | 405 | 1236 | 236 | 54 |

## Bad Buckets

| name | bucket | n | acc | good_recall | medium_recall | bad_recall | good_to_medium | medium_to_good | bad_to_good | bad_to_medium |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| featurefirst_top20_shift_stress_a050 | bad_core_nearboundary | 119 | 1.000000 | nan | nan | 1.000000 | 0 | 0 | 0 | 0 |
| featurefirst_top20_shift_stress_a050_seed18 | bad_core_nearboundary | 119 | 1.000000 | nan | nan | 1.000000 | 0 | 0 | 0 | 0 |
| featurefirst_top20_shift_stress_a050_seed21 | bad_core_nearboundary | 119 | 1.000000 | nan | nan | 1.000000 | 0 | 0 | 0 | 0 |
| predtop20_sqiquery_subject111_sparseevent_v5_badonly | bad_core_nearboundary | 119 | 1.000000 | nan | nan | 1.000000 | 0 | 0 | 0 | 0 |
| equal_featurefirst_sparsebad | bad_core_nearboundary | 119 | 1.000000 | nan | nan | 1.000000 | 0 | 0 | 0 | 0 |
| equal_old_featurefirst_sparsebad | bad_core_nearboundary | 119 | 1.000000 | nan | nan | 1.000000 | 0 | 0 | 0 | 0 |
| equal_featurefirst_a050_seeds | bad_core_nearboundary | 119 | 1.000000 | nan | nan | 1.000000 | 0 | 0 | 0 | 0 |
| equal_old_plus_featurefirst_seeds | bad_core_nearboundary | 119 | 1.000000 | nan | nan | 1.000000 | 0 | 0 | 0 | 0 |
| equal_all_structural | bad_core_nearboundary | 119 | 1.000000 | nan | nan | 1.000000 | 0 | 0 | 0 | 0 |
| equal_top2_old_featurefirst | bad_core_nearboundary | 119 | 0.991597 | nan | nan | 0.991597 | 0 | 0 | 0 | 1 |
| equal_hardrec_goodguard_badlite | bad_core_nearboundary | 119 | 0.991597 | nan | nan | 0.991597 | 0 | 0 | 0 | 1 |
| equal_top4_old_p20_burstdrop_sparse | bad_core_nearboundary | 119 | 0.991597 | nan | nan | 0.991597 | 0 | 0 | 0 | 1 |
| equal_hardrec_badlite | bad_core_nearboundary | 119 | 0.983193 | nan | nan | 0.983193 | 0 | 0 | 0 | 2 |
| predtop20_sqiquery_thresholdtree_badguard_pretrain | bad_core_nearboundary | 119 | 0.974790 | nan | nan | 0.974790 | 0 | 0 | 0 | 3 |
| predtop20_sqiquery_subject111_shift_stress_pretrain | bad_core_nearboundary | 119 | 0.966387 | nan | nan | 0.966387 | 0 | 0 | 0 | 4 |
| equal_top2_old_qrsstress | bad_core_nearboundary | 119 | 0.966387 | nan | nan | 0.966387 | 0 | 0 | 0 | 4 |
| featurefirst_top20_shift_stress_a050_seed19 | bad_core_nearboundary | 119 | 0.957983 | nan | nan | 0.957983 | 0 | 0 | 0 | 5 |
| equal_hardrec_badlite_qrsfocus | bad_core_nearboundary | 119 | 0.957983 | nan | nan | 0.957983 | 0 | 0 | 0 | 5 |
| equal_hardrec_all_badlite | bad_core_nearboundary | 119 | 0.957983 | nan | nan | 0.957983 | 0 | 0 | 0 | 5 |
| featurefirst_top20_hardrec_goodguard_badlite_a050 | bad_core_nearboundary | 119 | 0.949580 | nan | nan | 0.949580 | 0 | 0 | 0 | 6 |
| equal_old_plus_structural | bad_core_nearboundary | 119 | 0.949580 | nan | nan | 0.949580 | 0 | 0 | 0 | 6 |
| equal_top3_old_p20_qrsstress | bad_core_nearboundary | 119 | 0.941176 | nan | nan | 0.941176 | 0 | 0 | 0 | 7 |
| featurefirst_top20_hardrec_a050 | bad_core_nearboundary | 119 | 0.932773 | nan | nan | 0.932773 | 0 | 0 | 0 | 8 |
| featurefirst_top20_hardrec_badlite_a050 | bad_core_nearboundary | 119 | 0.932773 | nan | nan | 0.932773 | 0 | 0 | 0 | 8 |
| predtop20_sqiquery_subject111_burstdrop_dual_p26 | bad_core_nearboundary | 119 | 0.924370 | nan | nan | 0.924370 | 0 | 0 | 0 | 9 |
| featurefirst_top20_shift_stress_a050_seed20 | bad_core_nearboundary | 119 | 0.899160 | nan | nan | 0.899160 | 0 | 0 | 0 | 12 |
| predtop20_sqiquery_subject111_impulsebad_dual_p20 | bad_core_nearboundary | 119 | 0.899160 | nan | nan | 0.899160 | 0 | 0 | 0 | 12 |
| predtop20_sqiquery_qrsstressv3_stress_pretrain | bad_core_nearboundary | 119 | 0.831933 | nan | nan | 0.831933 | 0 | 0 | 0 | 20 |
| predtop20_sqiquery_primtree_stress_teacher_pretrain | bad_core_nearboundary | 119 | 0.831933 | nan | nan | 0.831933 | 0 | 0 | 0 | 20 |
| equal_hardrec_qrslite | bad_core_nearboundary | 119 | 0.823529 | nan | nan | 0.823529 | 0 | 0 | 0 | 21 |
| featurefirst_top20_hardrec_qrslite_a050 | bad_core_nearboundary | 119 | 0.747899 | nan | nan | 0.747899 | 0 | 0 | 0 | 30 |
| featurefirst_top20_hardrec_qrsfocus_a050 | bad_core_nearboundary | 119 | 0.722689 | nan | nan | 0.722689 | 0 | 0 | 0 | 33 |
| predtop20_sqiquery_qrsstressv3_pretrain | bad_core_nearboundary | 119 | 0.436975 | nan | nan | 0.436975 | 0 | 0 | 0 | 67 |
| featurefirst_top20_shift_stress_a050 | bad_outlier_stress | 292 | 0.082192 | nan | nan | 0.082192 | 0 | 0 | 188 | 80 |
| featurefirst_top20_hardrec_badlite_a050 | bad_outlier_stress | 292 | 0.078767 | nan | nan | 0.078767 | 0 | 0 | 227 | 42 |
| featurefirst_top20_hardrec_qrsfocus_a050 | bad_outlier_stress | 292 | 0.068493 | nan | nan | 0.068493 | 0 | 0 | 188 | 84 |
| predtop20_sqiquery_subject111_sparseevent_v5_badonly | bad_outlier_stress | 292 | 0.065068 | nan | nan | 0.065068 | 0 | 0 | 194 | 79 |
| equal_featurefirst_sparsebad | bad_outlier_stress | 292 | 0.065068 | nan | nan | 0.065068 | 0 | 0 | 192 | 81 |
| equal_old_featurefirst_sparsebad | bad_outlier_stress | 292 | 0.054795 | nan | nan | 0.054795 | 0 | 0 | 193 | 83 |
| equal_hardrec_badlite_qrsfocus | bad_outlier_stress | 292 | 0.047945 | nan | nan | 0.047945 | 0 | 0 | 194 | 84 |
| equal_top2_old_featurefirst | bad_outlier_stress | 292 | 0.044521 | nan | nan | 0.044521 | 0 | 0 | 197 | 82 |
| featurefirst_top20_shift_stress_a050_seed18 | bad_outlier_stress | 292 | 0.041096 | nan | nan | 0.041096 | 0 | 0 | 217 | 63 |
| predtop20_sqiquery_subject111_shift_stress_pretrain | bad_outlier_stress | 292 | 0.037671 | nan | nan | 0.037671 | 0 | 0 | 205 | 76 |
| equal_top4_old_p20_burstdrop_sparse | bad_outlier_stress | 292 | 0.037671 | nan | nan | 0.037671 | 0 | 0 | 194 | 87 |
| featurefirst_top20_hardrec_goodguard_badlite_a050 | bad_outlier_stress | 292 | 0.034247 | nan | nan | 0.034247 | 0 | 0 | 203 | 79 |
| featurefirst_top20_shift_stress_a050_seed20 | bad_outlier_stress | 292 | 0.030822 | nan | nan | 0.030822 | 0 | 0 | 234 | 49 |
| equal_hardrec_all_badlite | bad_outlier_stress | 292 | 0.030822 | nan | nan | 0.030822 | 0 | 0 | 201 | 82 |
| featurefirst_top20_hardrec_qrslite_a050 | bad_outlier_stress | 292 | 0.027397 | nan | nan | 0.027397 | 0 | 0 | 215 | 69 |
| equal_hardrec_goodguard_badlite | bad_outlier_stress | 292 | 0.027397 | nan | nan | 0.027397 | 0 | 0 | 190 | 94 |
| equal_old_plus_featurefirst_seeds | bad_outlier_stress | 292 | 0.027397 | nan | nan | 0.027397 | 0 | 0 | 221 | 63 |
| equal_all_structural | bad_outlier_stress | 292 | 0.027397 | nan | nan | 0.027397 | 0 | 0 | 212 | 72 |
| equal_top2_old_qrsstress | bad_outlier_stress | 292 | 0.023973 | nan | nan | 0.023973 | 0 | 0 | 207 | 78 |
| equal_featurefirst_a050_seeds | bad_outlier_stress | 292 | 0.023973 | nan | nan | 0.023973 | 0 | 0 | 225 | 60 |
| predtop20_sqiquery_qrsstressv3_stress_pretrain | bad_outlier_stress | 292 | 0.020548 | nan | nan | 0.020548 | 0 | 0 | 195 | 91 |
| equal_hardrec_badlite | bad_outlier_stress | 292 | 0.020548 | nan | nan | 0.020548 | 0 | 0 | 200 | 86 |
| equal_hardrec_qrslite | bad_outlier_stress | 292 | 0.020548 | nan | nan | 0.020548 | 0 | 0 | 194 | 92 |
| predtop20_sqiquery_thresholdtree_badguard_pretrain | bad_outlier_stress | 292 | 0.017123 | nan | nan | 0.017123 | 0 | 0 | 236 | 51 |
| predtop20_sqiquery_subject111_impulsebad_dual_p20 | bad_outlier_stress | 292 | 0.013699 | nan | nan | 0.013699 | 0 | 0 | 190 | 98 |
| predtop20_sqiquery_subject111_burstdrop_dual_p26 | bad_outlier_stress | 292 | 0.013699 | nan | nan | 0.013699 | 0 | 0 | 200 | 88 |
| equal_top3_old_p20_qrsstress | bad_outlier_stress | 292 | 0.013699 | nan | nan | 0.013699 | 0 | 0 | 196 | 92 |
| equal_old_plus_structural | bad_outlier_stress | 292 | 0.013699 | nan | nan | 0.013699 | 0 | 0 | 219 | 69 |
| featurefirst_top20_hardrec_a050 | bad_outlier_stress | 292 | 0.006849 | nan | nan | 0.006849 | 0 | 0 | 169 | 121 |
| featurefirst_top20_shift_stress_a050_seed19 | bad_outlier_stress | 292 | 0.006849 | nan | nan | 0.006849 | 0 | 0 | 234 | 56 |
| featurefirst_top20_shift_stress_a050_seed21 | bad_outlier_stress | 292 | 0.006849 | nan | nan | 0.006849 | 0 | 0 | 223 | 67 |
| predtop20_sqiquery_qrsstressv3_pretrain | bad_outlier_stress | 292 | 0.000000 | nan | nan | 0.000000 | 0 | 0 | 224 | 68 |
| predtop20_sqiquery_primtree_stress_teacher_pretrain | bad_outlier_stress | 292 | 0.000000 | nan | nan | 0.000000 | 0 | 0 | 245 | 47 |

## Interpretation

- If fixed averaging beats every member, the candidates contain complementary waveform evidence.
- If it does not, the current waveform-only family is representation/coverage limited rather than merely checkpoint-limited.