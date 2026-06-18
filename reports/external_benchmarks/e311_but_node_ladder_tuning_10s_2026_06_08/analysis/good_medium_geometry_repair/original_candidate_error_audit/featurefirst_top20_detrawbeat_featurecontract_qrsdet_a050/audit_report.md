# Original Candidate Error Audit: featurefirst_top20_detrawbeat_featurecontract_qrsdet_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.765601 | 0.925000 | 0.676457 | 0.313869 | 268 | 1385 | 44 |
| raw | original_all_10s+ | 0.842942 | 0.836003 | 0.808431 | 0.934721 | 2787 | 1976 | 105 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.034247 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 44 |
| badcal | original_test_all_10s+ | 0.765011 | 0.924451 | 0.667194 | 0.406326 | 263 | 1346 | 34 |
| badcal | original_all_10s+ | 0.842275 | 0.835886 | 0.802503 | 0.942857 | 2772 | 1936 | 91 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.164384 | 0.000000 | 0.000000 | 0.164384 | 0 | 0 | 34 |

## Error Counts

- test errors raw: 1987
- bad outlier errors raw: 282
- bad core errors raw: 0
- good->medium raw: 268
- medium->good raw: 1385
- nonbad->bad raw: 52

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_detrawbeat_featurecontract_qrsdet_a050/original_error_waveform_panels.png)
