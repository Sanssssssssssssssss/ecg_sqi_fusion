# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.855019 | 0.894505 | 0.869182 | 0.352798 | 381 | 511 | 80 |
| raw | original_all_10s+ | 0.866519 | 0.822742 | 0.901487 | 0.937370 | 3017 | 973 | 144 |
| raw | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| raw | bad_outlier_stress | 0.106164 | 0.000000 | 0.000000 | 0.106164 | 0 | 0 | 75 |
| badcal | original_test_all_10s+ | 0.857497 | 0.894505 | 0.862178 | 0.479319 | 380 | 483 | 61 |
| badcal | original_all_10s+ | 0.867065 | 0.822742 | 0.898288 | 0.947209 | 3016 | 945 | 125 |
| badcal | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| badcal | bad_outlier_stress | 0.273973 | 0.000000 | 0.000000 | 0.273973 | 0 | 0 | 59 |

## Error Counts

- test errors raw: 1229
- bad outlier errors raw: 261
- bad core errors raw: 5
- good->medium raw: 381
- medium->good raw: 511
- nonbad->bad raw: 71

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050/original_error_waveform_panels.png)
