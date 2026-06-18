# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_eventcontract_guard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.858440 | 0.889835 | 0.885224 | 0.291971 | 400 | 488 | 98 |
| raw | original_all_10s+ | 0.866155 | 0.818459 | 0.909673 | 0.932450 | 3093 | 936 | 162 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.023973 | 0.000000 | 0.000000 | 0.023973 | 0 | 0 | 92 |
| badcal | original_test_all_10s+ | 0.860210 | 0.889835 | 0.874379 | 0.445255 | 395 | 457 | 72 |
| badcal | original_all_10s+ | 0.866428 | 0.818459 | 0.904309 | 0.944939 | 3082 | 905 | 134 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.219178 | 0.000000 | 0.000000 | 0.219178 | 0 | 0 | 72 |

## Error Counts

- test errors raw: 1200
- bad outlier errors raw: 285
- bad core errors raw: 6
- good->medium raw: 400
- medium->good raw: 488
- nonbad->bad raw: 21

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_eventcontract_guard_a050/original_error_waveform_panels.png)
