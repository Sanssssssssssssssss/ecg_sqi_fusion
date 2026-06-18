# Bad-Stress Primitive Rule Search (Report Only)

- Candidate: `featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050`
- Badcal threshold: `0.480000`
- Primitive bank: `qrs_stress_v5`
- Original BUT is used only as a non-test/test diagnostic probe here, not for checkpoint selection.

## Baseline Original Test

| bad_outlier_stress_bad_rate | n | acc | recall_good | recall_medium | recall_bad |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 8477 | 0.864103 | 0.88956 | 0.864437 | 0.635036 |

## Top Boost Rules

_No rows._

## Top Veto Rules

| feature | direction | threshold_z | train_score | test_nonbad_falsebad_removed | test_bad_hit_lost | test_acc | test_acc_gain | test_recall_good | test_recall_medium | test_recall_bad | test_bad_recall_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ch1_sparse_event_bank_07 | low | -0.428972 | 14 | 160 | 53 | 0.876135 | 0.0120326 | 0.88956 | 0.899458 | 0.506083 | -0.128954 |
| ch1_sparse_event_bank_10 | high | 1.20862 | 14 | 160 | 53 | 0.876135 | 0.0120326 | 0.88956 | 0.899458 | 0.506083 | -0.128954 |
| ch0_sparse_event_bank_06 | low | -2.56399 | 14 | 156 | 51 | 0.875899 | 0.0117966 | 0.88956 | 0.898554 | 0.510949 | -0.124088 |
| ch2_stress_bank_a_22 | low | -2.56399 | 14 | 156 | 51 | 0.875899 | 0.0117966 | 0.88956 | 0.898554 | 0.510949 | -0.124088 |
| ch1_sparse_event_bank_04 | low | -0.43547 | 14 | 147 | 43 | 0.875782 | 0.0116787 | 0.88956 | 0.896521 | 0.530414 | -0.104623 |
| ch1_sparse_event_bank_04 | low | -0.419454 | 14 | 167 | 66 | 0.875428 | 0.0113248 | 0.88956 | 0.901039 | 0.474453 | -0.160584 |
| ch1_sparse_event_bank_06 | low | -0.359971 | 14 | 153 | 52 | 0.875428 | 0.0113248 | 0.88956 | 0.897876 | 0.508516 | -0.126521 |
| ch1_stress_bank_b_21 | low | -0.956079 | 14 | 154 | 54 | 0.87531 | 0.0112068 | 0.88956 | 0.898102 | 0.50365 | -0.131387 |
| ch1_sparse_event_bank_06 | low | -0.377118 | 14 | 147 | 47 | 0.87531 | 0.0112068 | 0.88956 | 0.896521 | 0.520681 | -0.114355 |
| ch1_sparse_event_bank_06 | low | -0.342025 | 14 | 158 | 58 | 0.87531 | 0.0112068 | 0.88956 | 0.899006 | 0.493917 | -0.141119 |
| ch1_stress_bank_a_15 | low | -1.12914 | 14 | 165 | 66 | 0.875192 | 0.0110888 | 0.88956 | 0.900587 | 0.474453 | -0.160584 |
| ch1_stress_bank_b_04 | high | 0.72006 | 14 | 151 | 52 | 0.875192 | 0.0110888 | 0.88956 | 0.897424 | 0.508516 | -0.126521 |
| ch1_sparse_event_bank_06 | low | -0.349656 | 14 | 155 | 56 | 0.875192 | 0.0110888 | 0.88956 | 0.898328 | 0.498783 | -0.136253 |
| ch0_sparse_event_bank_05 | low | -1.00144 | 14 | 140 | 43 | 0.875074 | 0.0109709 | 0.88956 | 0.895165 | 0.530414 | -0.104623 |
| ch1_sparse_event_bank_03 | low | -1.00144 | 14 | 140 | 43 | 0.875074 | 0.0109709 | 0.88956 | 0.895165 | 0.530414 | -0.104623 |
| ch1_sparse_event_bank_06 | low | -0.335652 | 14 | 159 | 61 | 0.875074 | 0.0109709 | 0.88956 | 0.899232 | 0.486618 | -0.148418 |
| ch2_stress_bank_a_21 | low | -1.00144 | 14 | 140 | 43 | 0.875074 | 0.0109709 | 0.88956 | 0.895165 | 0.530414 | -0.104623 |
| ch1_stress_bank_b_04 | high | 0.598482 | 14 | 156 | 59 | 0.874956 | 0.0108529 | 0.88956 | 0.898554 | 0.491484 | -0.143552 |
| ch1_waveform_basic_01 | low | -0.787466 | 14 | 145 | 50 | 0.87472 | 0.010617 | 0.88956 | 0.896069 | 0.513382 | -0.121655 |
| ch1_stress_bank_b_04 | high | 0.327512 | 14 | 167 | 72 | 0.87472 | 0.010617 | 0.88956 | 0.901039 | 0.459854 | -0.175182 |