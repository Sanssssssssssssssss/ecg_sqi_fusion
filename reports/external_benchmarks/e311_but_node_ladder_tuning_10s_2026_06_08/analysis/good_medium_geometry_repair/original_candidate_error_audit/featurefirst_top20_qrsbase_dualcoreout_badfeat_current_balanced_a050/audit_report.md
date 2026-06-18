# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.857379 | 0.889835 | 0.880479 | 0.321168 | 399 | 481 | 91 |
| raw | original_all_10s+ | 0.865912 | 0.818459 | 0.907697 | 0.934910 | 3091 | 929 | 155 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.065068 | 0.000000 | 0.000000 | 0.065068 | 0 | 0 | 85 |
| badcal | original_test_all_10s+ | 0.861508 | 0.889835 | 0.869634 | 0.523114 | 395 | 439 | 60 |
| badcal | original_all_10s+ | 0.866883 | 0.818459 | 0.902710 | 0.950993 | 3086 | 887 | 122 |
| badcal | bad_core_nearboundary | 0.966387 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 |
| badcal | bad_outlier_stress | 0.342466 | 0.000000 | 0.000000 | 0.342466 | 0 | 0 | 56 |

## Error Counts

- test errors raw: 1209
- bad outlier errors raw: 273
- bad core errors raw: 6
- good->medium raw: 399
- medium->good raw: 481
- nonbad->bad raw: 50

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050/original_error_waveform_panels.png)
