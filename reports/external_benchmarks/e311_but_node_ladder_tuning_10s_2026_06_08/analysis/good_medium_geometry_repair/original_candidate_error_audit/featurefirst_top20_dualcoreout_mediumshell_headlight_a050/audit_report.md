# Original Candidate Error Audit: featurefirst_top20_dualcoreout_mediumshell_headlight_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.822343 | 0.916209 | 0.795526 | 0.279805 | 304 | 894 | 79 |
| raw | original_all_10s+ | 0.871617 | 0.885877 | 0.819063 | 0.931315 | 1944 | 1909 | 144 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 73 |
| badcal | original_test_all_10s+ | 0.822107 | 0.916209 | 0.786037 | 0.377129 | 301 | 877 | 54 |
| badcal | original_all_10s+ | 0.871495 | 0.885877 | 0.814546 | 0.939640 | 1940 | 1892 | 116 |
| badcal | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| badcal | bad_outlier_stress | 0.126712 | 0.000000 | 0.000000 | 0.126712 | 0 | 0 | 53 |

## Error Counts

- test errors raw: 1506
- bad outlier errors raw: 290
- bad core errors raw: 6
- good->medium raw: 304
- medium->good raw: 894
- nonbad->bad raw: 12

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_dualcoreout_mediumshell_headlight_a050/original_error_waveform_panels.png)
