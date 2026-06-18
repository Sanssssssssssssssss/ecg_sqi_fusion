# Original Candidate Error Audit: featurefirst_top20_qrsbase_primauxres_current_badguard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.799929 | 0.916484 | 0.750565 | 0.299270 | 297 | 1039 | 64 |
| raw | original_all_10s+ | 0.865063 | 0.871208 | 0.821415 | 0.933018 | 2183 | 1823 | 127 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.034247 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 58 |
| badcal | original_test_all_10s+ | 0.801345 | 0.916484 | 0.741527 | 0.425791 | 296 | 989 | 44 |
| badcal | original_all_10s+ | 0.865184 | 0.871208 | 0.816805 | 0.943046 | 2178 | 1773 | 107 |
| badcal | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| badcal | bad_outlier_stress | 0.198630 | 0.000000 | 0.000000 | 0.198630 | 0 | 0 | 42 |

## Error Counts

- test errors raw: 1696
- bad outlier errors raw: 282
- bad core errors raw: 6
- good->medium raw: 297
- medium->good raw: 1039
- nonbad->bad raw: 72

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_primauxres_current_badguard_a050/original_error_waveform_panels.png)
