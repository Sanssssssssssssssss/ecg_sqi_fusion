# Original Candidate Error Audit: featurefirst_top20_detrawbeat_artifact_contract_badguard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.760411 | 0.929945 | 0.662901 | 0.309002 | 253 | 1455 | 36 |
| raw | original_all_10s+ | 0.840848 | 0.834888 | 0.804008 | 0.934153 | 2811 | 2036 | 98 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.027397 | 0.000000 | 0.000000 | 0.027397 | 0 | 0 | 36 |
| badcal | original_test_all_10s+ | 0.756518 | 0.929396 | 0.645278 | 0.423358 | 245 | 1406 | 30 |
| badcal | original_all_10s+ | 0.839149 | 0.834771 | 0.793941 | 0.944182 | 2791 | 1987 | 87 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.188356 | 0.000000 | 0.000000 | 0.188356 | 0 | 0 | 30 |

## Error Counts

- test errors raw: 2031
- bad outlier errors raw: 284
- bad core errors raw: 0
- good->medium raw: 253
- medium->good raw: 1455
- nonbad->bad raw: 39

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_detrawbeat_artifact_contract_badguard_a050/original_error_waveform_panels.png)
