# Original Candidate Error Audit: featurefirst_wavecomp_dualcoreout_encoderlite_mediumguard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.789312 | 0.923626 | 0.708992 | 0.464720 | 272 | 1177 | 39 |
| raw | original_all_10s+ | 0.856748 | 0.855131 | 0.814923 | 0.946074 | 2461 | 1848 | 102 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.267123 | 0.000000 | 0.000000 | 0.267123 | 0 | 0 | 33 |
| badcal | original_test_all_10s+ | 0.789902 | 0.923626 | 0.708992 | 0.476886 | 272 | 1176 | 39 |
| badcal | original_all_10s+ | 0.856900 | 0.855131 | 0.814923 | 0.947020 | 2461 | 1847 | 102 |
| badcal | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| badcal | bad_outlier_stress | 0.284247 | 0.000000 | 0.000000 | 0.284247 | 0 | 0 | 33 |

## Error Counts

- test errors raw: 1786
- bad outlier errors raw: 214
- bad core errors raw: 6
- good->medium raw: 272
- medium->good raw: 1177
- nonbad->bad raw: 117

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_wavecomp_dualcoreout_encoderlite_mediumguard_a050/original_error_waveform_panels.png)
