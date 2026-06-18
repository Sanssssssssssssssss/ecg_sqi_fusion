# Original Candidate Error Audit: featurefirst_top23_shift_stress_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.821281 | 0.874451 | 0.828739 | 0.270073 | 455 | 737 | 118 |
| raw | original_all_10s+ | 0.819760 | 0.730857 | 0.907226 | 0.930558 | 4584 | 957 | 183 |
| raw | bad_core_nearboundary | 0.907563 | 0.000000 | 0.000000 | 0.907563 | 0 | 0 | 11 |
| raw | bad_outlier_stress | 0.010274 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 107 |
| badcal | original_test_all_10s+ | 0.783650 | 0.855220 | 0.759828 | 0.406326 | 420 | 539 | 78 |
| badcal | original_all_10s+ | 0.806712 | 0.722701 | 0.873824 | 0.942668 | 4489 | 750 | 136 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.164384 | 0.000000 | 0.000000 | 0.164384 | 0 | 0 | 78 |

## Error Counts

- test errors raw: 1515
- bad outlier errors raw: 289
- bad core errors raw: 11
- good->medium raw: 455
- medium->good raw: 737
- nonbad->bad raw: 23

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top23_shift_stress_a050/original_error_waveform_panels.png)
