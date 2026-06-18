# Original Candidate Error Audit: featurefirst_top20_hardrec_goodguard_badlite_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.826590 | 0.879945 | 0.831676 | 0.299270 | 425 | 684 | 85 |
| raw | original_all_10s+ | 0.836449 | 0.765241 | 0.903933 | 0.930369 | 3975 | 946 | 163 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.034247 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 79 |
| badcal | original_test_all_10s+ | 0.795800 | 0.870879 | 0.771351 | 0.394161 | 404 | 561 | 62 |
| badcal | original_all_10s+ | 0.824402 | 0.758376 | 0.871942 | 0.941722 | 3867 | 811 | 120 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.147260 | 0.000000 | 0.000000 | 0.147260 | 0 | 0 | 62 |

## Error Counts

- test errors raw: 1470
- bad outlier errors raw: 282
- bad core errors raw: 6
- good->medium raw: 425
- medium->good raw: 684
- nonbad->bad raw: 73

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_hardrec_goodguard_badlite_a050/original_error_waveform_panels.png)
