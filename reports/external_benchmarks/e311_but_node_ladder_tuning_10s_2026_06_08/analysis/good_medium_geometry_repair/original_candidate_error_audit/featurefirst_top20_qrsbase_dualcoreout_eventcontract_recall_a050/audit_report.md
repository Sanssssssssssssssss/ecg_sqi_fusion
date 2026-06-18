# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_eventcontract_recall_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.858323 | 0.889835 | 0.884094 | 0.301703 | 399 | 486 | 95 |
| raw | original_all_10s+ | 0.866094 | 0.818459 | 0.909108 | 0.933207 | 3091 | 934 | 159 |
| raw | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| raw | bad_outlier_stress | 0.034247 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 90 |
| badcal | original_test_all_10s+ | 0.860918 | 0.889835 | 0.872571 | 0.479319 | 395 | 454 | 67 |
| badcal | original_all_10s+ | 0.866671 | 0.818459 | 0.903745 | 0.947588 | 3086 | 902 | 129 |
| badcal | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| badcal | bad_outlier_stress | 0.270548 | 0.000000 | 0.000000 | 0.270548 | 0 | 0 | 66 |

## Error Counts

- test errors raw: 1201
- bad outlier errors raw: 282
- bad core errors raw: 5
- good->medium raw: 399
- medium->good raw: 486
- nonbad->bad raw: 29

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_eventcontract_recall_a050/original_error_waveform_panels.png)
