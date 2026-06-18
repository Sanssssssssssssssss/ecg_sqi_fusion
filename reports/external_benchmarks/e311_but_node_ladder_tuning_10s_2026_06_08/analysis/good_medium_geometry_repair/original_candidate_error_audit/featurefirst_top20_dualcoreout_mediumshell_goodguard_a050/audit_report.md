# Original Candidate Error Audit: featurefirst_top20_dualcoreout_mediumshell_goodguard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.808895 | 0.920330 | 0.765929 | 0.284672 | 288 | 1020 | 69 |
| raw | original_all_10s+ | 0.869644 | 0.896262 | 0.796011 | 0.931883 | 1765 | 2148 | 133 |
| raw | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| raw | bad_outlier_stress | 0.010274 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 64 |
| badcal | original_test_all_10s+ | 0.809013 | 0.920330 | 0.757569 | 0.377129 | 286 | 1003 | 46 |
| badcal | original_all_10s+ | 0.869614 | 0.896262 | 0.792059 | 0.939640 | 1763 | 2131 | 107 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.123288 | 0.000000 | 0.000000 | 0.123288 | 0 | 0 | 46 |

## Error Counts

- test errors raw: 1620
- bad outlier errors raw: 289
- bad core errors raw: 5
- good->medium raw: 288
- medium->good raw: 1020
- nonbad->bad raw: 18

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_dualcoreout_mediumshell_goodguard_a050/original_error_waveform_panels.png)
