# Event-Contract Bad Gap Compare

- Current: `featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050` badcal threshold `0.480000`
- Event-contract: `featurefirst_top20_qrsbase_dualcoreout_eventcontract_recall_a050` badcal threshold `0.130000`
- Primitive bank: `qrs_stress_v5`

## Set Counts

| set | n | good | medium | bad | current_prob_bad_mean | event_prob_bad_mean | current_prob_bad_p90 | event_prob_bad_p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| both_hit | 79 | 0 | 0 | 79 | 0.740668 | 0.263346 | 0.875938 | 0.42134 |
| current_hit_event_miss | 67 | 0 | 0 | 67 | 0.596527 | 0.0864624 | 0.723692 | 0.120203 |
| event_hit_current_miss | 0 | 0 | 0 | 0 | nan | nan | nan | nan |
| both_miss | 146 | 0 | 0 | 146 | 0.128159 | 0.0297105 | 0.364646 | 0.0588806 |
| current_false_bad_nonbad | 181 | 5 | 176 | 0 | 0.732357 | 0.235229 | 0.935048 | 0.600156 |
| event_false_bad_nonbad | 116 | 6 | 110 | 0 | 0.780426 | 0.329357 | 0.957521 | 0.741061 |

## Top Primitive Gaps

| contrast | primitive | mean_z_current_hit_event_miss | mean_z_both_miss | delta_z | mean_z_both_hit |
| --- | --- | --- | --- | --- | --- |
| current_hit_event_miss_minus_both_miss | ch2_stress_bank_06 | 4.40593 | 3.07134 | 1.33458 | nan |
| current_hit_event_miss_minus_both_miss | ch2_stress_bank_12 | 3.8016 | 2.65108 | 1.15053 | nan |
| current_hit_event_miss_minus_both_miss | ch0_sparse_event_bank_29 | 0.418791 | 1.55466 | -1.13587 | nan |
| current_hit_event_miss_minus_both_miss | ch1_atlas_00 | 0.309879 | 1.42373 | -1.11385 | nan |
| current_hit_event_miss_minus_both_miss | ch2_stress_bank_05 | 3.95687 | 2.87632 | 1.08055 | nan |
| current_hit_event_miss_minus_both_miss | ch0_sparse_event_bank_17 | 1.06024 | 2.12956 | -1.06933 | nan |
| current_hit_event_miss_minus_both_miss | ch0_detector_agreement_bank_15 | 1.06024 | 2.12956 | -1.06933 | nan |
| current_hit_event_miss_minus_both_miss | ch2_qrs_visibility_bank_09 | 1.06024 | 2.12956 | -1.06933 | nan |
| current_hit_event_miss_minus_both_miss | ch0_atlas_14 | 1.06024 | 2.12956 | -1.06933 | nan |
| current_hit_event_miss_minus_both_miss | ch0_baseline_frequency_bank_04 | 2.42827 | 1.36687 | 1.06141 | nan |
| current_hit_event_miss_minus_both_miss | ch2_atlas_03 | 2.42827 | 1.36687 | 1.06141 | nan |
| current_hit_event_miss_minus_both_miss | ch2_baseline_frequency_bank_04 | 0.751101 | 1.72804 | -0.976942 | nan |
| current_hit_event_miss_minus_both_miss | ch0_baseline_frequency_bank_07 | 2.55706 | 1.58069 | 0.976374 | nan |
| current_hit_event_miss_minus_both_miss | ch0_atlas_22 | 2.55706 | 1.58069 | 0.976374 | nan |
| current_hit_event_miss_minus_both_miss | ch2_atlas_06 | 2.55706 | 1.58069 | 0.976374 | nan |
| current_hit_event_miss_minus_both_miss | ch0_detector_agreement_bank_02 | 2.55706 | 1.58069 | 0.976374 | nan |
| current_hit_event_miss_minus_both_miss | ch2_qrs_visibility_bank_14 | 2.55412 | 1.578 | 0.976116 | nan |
| current_hit_event_miss_minus_both_miss | ch0_sparse_event_bank_22 | 2.55412 | 1.578 | 0.976116 | nan |
| current_hit_event_miss_minus_both_miss | ch2_baseline_frequency_bank_07 | 0.753107 | 1.72688 | -0.97377 | nan |
| current_hit_event_miss_minus_both_miss | ch2_baseline_frequency_bank_10 | 0.731159 | 1.69029 | -0.959131 | nan |
