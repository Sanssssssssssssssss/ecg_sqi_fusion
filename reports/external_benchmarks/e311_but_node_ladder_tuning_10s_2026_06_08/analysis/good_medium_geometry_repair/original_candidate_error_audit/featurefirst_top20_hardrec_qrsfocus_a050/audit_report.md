# Original Candidate Error Audit: featurefirst_top20_hardrec_qrsfocus_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.812080 | 0.874451 | 0.812246 | 0.257908 | 443 | 737 | 117 |
| raw | original_all_10s+ | 0.838937 | 0.786892 | 0.877399 | 0.929423 | 3599 | 1181 | 183 |
| raw | bad_core_nearboundary | 0.722689 | 0.000000 | 0.000000 | 0.722689 | 0 | 0 | 33 |
| raw | bad_outlier_stress | 0.068493 | 0.000000 | 0.000000 | 0.068493 | 0 | 0 | 84 |
| badcal | original_test_all_10s+ | 0.761472 | 0.841484 | 0.725034 | 0.445255 | 391 | 555 | 59 |
| badcal | original_all_10s+ | 0.820154 | 0.772575 | 0.834023 | 0.945695 | 3449 | 989 | 117 |
| badcal | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| badcal | bad_outlier_stress | 0.239726 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 53 |

## Error Counts

- test errors raw: 1593
- bad outlier errors raw: 272
- bad core errors raw: 33
- good->medium raw: 443
- medium->good raw: 737
- nonbad->bad raw: 108

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_hardrec_qrsfocus_a050/original_error_waveform_panels.png)
