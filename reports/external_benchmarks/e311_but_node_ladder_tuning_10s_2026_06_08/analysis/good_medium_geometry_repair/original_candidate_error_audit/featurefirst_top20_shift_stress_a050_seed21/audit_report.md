# Original Candidate Error Audit: featurefirst_top20_shift_stress_a050_seed21

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.790846 | 0.906044 | 0.742205 | 0.294404 | 342 | 1131 | 67 |
| raw | original_all_10s+ | 0.827467 | 0.774042 | 0.860839 | 0.932640 | 3850 | 1462 | 131 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 67 |
| badcal | original_test_all_10s+ | 0.784122 | 0.901923 | 0.727971 | 0.345499 | 339 | 1023 | 58 |
| badcal | original_all_10s+ | 0.823067 | 0.770228 | 0.850583 | 0.938127 | 3841 | 1343 | 115 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.078767 | 0.000000 | 0.000000 | 0.078767 | 0 | 0 | 58 |

## Error Counts

- test errors raw: 1773
- bad outlier errors raw: 290
- bad core errors raw: 0
- good->medium raw: 342
- medium->good raw: 1131
- nonbad->bad raw: 10

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_shift_stress_a050_seed21/original_error_waveform_panels.png)
