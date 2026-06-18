# Original Candidate Error Audit: featurefirst_top20_shift_stress_a025

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.790964 | 0.890110 | 0.756439 | 0.284672 | 398 | 1058 | 59 |
| raw | original_all_10s+ | 0.809473 | 0.734378 | 0.870342 | 0.929234 | 4523 | 1345 | 125 |
| raw | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| raw | bad_outlier_stress | 0.010274 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 54 |
| badcal | original_test_all_10s+ | 0.784122 | 0.885989 | 0.741075 | 0.345499 | 398 | 923 | 49 |
| badcal | original_all_10s+ | 0.805377 | 0.729801 | 0.860933 | 0.937370 | 4508 | 1195 | 106 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.078767 | 0.000000 | 0.000000 | 0.078767 | 0 | 0 | 49 |

## Error Counts

- test errors raw: 1772
- bad outlier errors raw: 289
- bad core errors raw: 5
- good->medium raw: 398
- medium->good raw: 1058
- nonbad->bad raw: 22

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_shift_stress_a025/original_error_waveform_panels.png)
