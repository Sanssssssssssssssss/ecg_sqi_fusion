# Original Candidate Error Audit: featurefirst_top20_qrsbase_tailfocus_goodguard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.814675 | 0.915934 | 0.780163 | 0.289538 | 305 | 947 | 62 |
| raw | original_all_10s+ | 0.846705 | 0.802617 | 0.874953 | 0.932072 | 3362 | 1297 | 127 |
| raw | bad_core_nearboundary | 0.974790 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 3 |
| raw | bad_outlier_stress | 0.010274 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 59 |
| badcal | original_test_all_10s+ | 0.820101 | 0.915659 | 0.775644 | 0.452555 | 301 | 889 | 48 |
| badcal | original_all_10s+ | 0.848192 | 0.802558 | 0.872695 | 0.946074 | 3357 | 1239 | 107 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.229452 | 0.000000 | 0.000000 | 0.229452 | 0 | 0 | 48 |

## Error Counts

- test errors raw: 1571
- bad outlier errors raw: 289
- bad core errors raw: 3
- good->medium raw: 305
- medium->good raw: 947
- nonbad->bad raw: 27

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_tailfocus_goodguard_a050/original_error_waveform_panels.png)
