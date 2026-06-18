# PredTop20 Bad Threshold Frontier

Report-only diagnostic on original test. Thresholds are not selection criteria.

## Best Acc
| candidate | threshold | acc | good_recall | medium_recall | bad_recall | bad_outlier_stress_recall | false_bad_nonbad |
| --- | --- | --- | --- | --- | --- | --- | --- |
| predtop20_sqiquery_boundary_pretrain | 0.700000 | 0.830954 | 0.830769 | 0.879349 | 0.311436 | 0.030822 | 101 |
| predtop20_sqiquery_balancedguard_pretrain | 0.970000 | 0.795447 | 0.888187 | 0.765251 | 0.299270 | 0.020548 | 83 |
| predtop20_sqiquery_badguardlite_pretrain | 0.530000 | 0.789548 | 0.884066 | 0.756665 | 0.306569 | 0.023973 | 142 |

## Best Stress Recall
| candidate | threshold | acc | good_recall | medium_recall | bad_recall | bad_outlier_stress_recall | false_bad_nonbad |
| --- | --- | --- | --- | --- | --- | --- | --- |
| predtop20_sqiquery_balancedguard_pretrain | 0.010000 | 0.672644 | 0.815110 | 0.553999 | 0.688564 | 0.561644 | 1922 |
| predtop20_sqiquery_badguardlite_pretrain | 0.010000 | 0.719948 | 0.805495 | 0.654993 | 0.661800 | 0.523973 | 1452 |
| predtop20_sqiquery_boundary_pretrain | 0.010000 | 0.741536 | 0.779945 | 0.723226 | 0.598540 | 0.434932 | 1278 |

## Balanced Diagnostic Score
| candidate | threshold | acc | good_recall | medium_recall | bad_recall | bad_outlier_stress_recall | false_bad_nonbad | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predtop20_sqiquery_boundary_pretrain | 0.700000 | 0.830954 | 0.830769 | 0.879349 | 0.311436 | 0.030822 | 101 | 0.825957 |
| predtop20_sqiquery_balancedguard_pretrain | 0.960000 | 0.795447 | 0.888187 | 0.765251 | 0.299270 | 0.020548 | 83 | 0.790861 |
| predtop20_sqiquery_badguardlite_pretrain | 0.520000 | 0.789548 | 0.884066 | 0.756665 | 0.306569 | 0.023973 | 142 | 0.780585 |

## p_bad Distributions
| candidate | group | n | p_bad_q05 | p_bad_q50 | p_bad_q95 | p_bad_mean |
| --- | --- | --- | --- | --- | --- | --- |
| predtop20_sqiquery_boundary_pretrain | test_bad_core_nearboundary | 119 | 0.810057 | 0.949500 | 0.975192 | 0.931587 |
| predtop20_sqiquery_boundary_pretrain | test_bad_outlier_stress | 292 | 0.001589 | 0.008254 | 0.183722 | 0.044899 |
| predtop20_sqiquery_boundary_pretrain | test_nonbad | 8066 | 0.000812 | 0.002694 | 0.068051 | 0.019515 |
| predtop20_sqiquery_balancedguard_pretrain | test_bad_core_nearboundary | 119 | 0.738468 | 0.974802 | 0.985747 | 0.933262 |
| predtop20_sqiquery_balancedguard_pretrain | test_bad_outlier_stress | 292 | 0.002204 | 0.011602 | 0.141541 | 0.039927 |
| predtop20_sqiquery_balancedguard_pretrain | test_nonbad | 8066 | 0.000957 | 0.004807 | 0.075923 | 0.020588 |
| predtop20_sqiquery_badguardlite_pretrain | test_bad_core_nearboundary | 119 | 0.867960 | 0.960876 | 0.979668 | 0.947985 |
| predtop20_sqiquery_badguardlite_pretrain | test_bad_outlier_stress | 292 | 0.002308 | 0.010633 | 0.174305 | 0.046641 |
| predtop20_sqiquery_badguardlite_pretrain | test_nonbad | 8066 | 0.000841 | 0.002323 | 0.101214 | 0.024210 |