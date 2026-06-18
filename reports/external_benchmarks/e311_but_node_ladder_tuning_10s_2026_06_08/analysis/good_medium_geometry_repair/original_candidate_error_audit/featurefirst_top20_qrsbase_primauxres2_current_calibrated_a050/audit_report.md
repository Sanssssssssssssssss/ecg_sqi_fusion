# Original Candidate Error Audit: featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.820573 | 0.913462 | 0.792363 | 0.301703 | 312 | 888 | 69 |
| raw | original_all_10s+ | 0.870555 | 0.872910 | 0.835623 | 0.933207 | 2162 | 1710 | 133 |
| raw | bad_core_nearboundary | 0.974790 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 3 |
| raw | bad_outlier_stress | 0.027397 | 0.000000 | 0.000000 | 0.027397 | 0 | 0 | 66 |
| badcal | original_test_all_10s+ | 0.822107 | 0.913462 | 0.786941 | 0.391727 | 311 | 861 | 50 |
| badcal | original_all_10s+ | 0.870919 | 0.872910 | 0.832894 | 0.940965 | 2160 | 1683 | 111 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.143836 | 0.000000 | 0.000000 | 0.143836 | 0 | 0 | 50 |

## Error Counts

- test errors raw: 1521
- bad outlier errors raw: 284
- bad core errors raw: 3
- good->medium raw: 312
- medium->good raw: 888
- nonbad->bad raw: 34

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050/original_error_waveform_panels.png)
