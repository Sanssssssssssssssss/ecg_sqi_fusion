# Original Candidate Error Audit: featurefirst_quality20_currentbest_headreset_badkeep_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.786245 | 0.925000 | 0.717126 | 0.301703 | 269 | 1222 | 40 |
| raw | original_all_10s+ | 0.862119 | 0.878014 | 0.801280 | 0.933207 | 2074 | 2074 | 103 |
| raw | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| raw | bad_outlier_stress | 0.023973 | 0.000000 | 0.000000 | 0.023973 | 0 | 0 | 38 |
| badcal | original_test_all_10s+ | 0.786363 | 0.924725 | 0.705377 | 0.433090 | 267 | 1155 | 25 |
| badcal | original_all_10s+ | 0.862059 | 0.877897 | 0.795540 | 0.944749 | 2069 | 2007 | 83 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.202055 | 0.000000 | 0.000000 | 0.202055 | 0 | 0 | 25 |

## Error Counts

- test errors raw: 1812
- bad outlier errors raw: 285
- bad core errors raw: 2
- good->medium raw: 269
- medium->good raw: 1222
- nonbad->bad raw: 34

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_quality20_currentbest_headreset_badkeep_a050/original_error_waveform_panels.png)
