# Original Candidate Error Audit: featurefirst_top20_qrsbase_primauxres2_current_badguard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.819158 | 0.914286 | 0.788974 | 0.301703 | 308 | 904 | 68 |
| raw | original_all_10s+ | 0.870524 | 0.874201 | 0.833459 | 0.933207 | 2139 | 1734 | 132 |
| raw | bad_core_nearboundary | 0.974790 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 3 |
| raw | bad_outlier_stress | 0.027397 | 0.000000 | 0.000000 | 0.027397 | 0 | 0 | 65 |
| badcal | original_test_all_10s+ | 0.820455 | 0.914286 | 0.783778 | 0.384428 | 308 | 880 | 49 |
| badcal | original_all_10s+ | 0.870828 | 0.874201 | 0.830824 | 0.940397 | 2138 | 1710 | 110 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.133562 | 0.000000 | 0.000000 | 0.133562 | 0 | 0 | 49 |

## Error Counts

- test errors raw: 1533
- bad outlier errors raw: 284
- bad core errors raw: 3
- good->medium raw: 308
- medium->good raw: 904
- nonbad->bad raw: 34

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_primauxres2_current_badguard_a050/original_error_waveform_panels.png)
