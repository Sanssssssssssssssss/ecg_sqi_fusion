# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_badfeat_current_guard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.856907 | 0.889835 | 0.876864 | 0.350365 | 398 | 474 | 84 |
| raw | original_all_10s+ | 0.865760 | 0.818459 | 0.906097 | 0.937181 | 3090 | 922 | 148 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.106164 | 0.000000 | 0.000000 | 0.106164 | 0 | 0 | 78 |
| badcal | original_test_all_10s+ | 0.861154 | 0.889835 | 0.868730 | 0.525547 | 397 | 437 | 54 |
| badcal | original_all_10s+ | 0.866731 | 0.818459 | 0.902333 | 0.950804 | 3089 | 885 | 118 |
| badcal | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| badcal | bad_outlier_stress | 0.339041 | 0.000000 | 0.000000 | 0.339041 | 0 | 0 | 52 |

## Error Counts

- test errors raw: 1213
- bad outlier errors raw: 261
- bad core errors raw: 6
- good->medium raw: 398
- medium->good raw: 474
- nonbad->bad raw: 74

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_badfeat_current_guard_a050/original_error_waveform_panels.png)
