# Original Candidate Error Audit: predtop20_sqiquery_subject111_shift_stress_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.823051 | 0.851923 | 0.847266 | 0.306569 | 531 | 635 | 80 |
| raw | original_all_10s+ | 0.824948 | 0.736901 | 0.912307 | 0.933207 | 4471 | 882 | 146 |
| raw | bad_core_nearboundary | 0.966387 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 |
| raw | bad_outlier_stress | 0.037671 | 0.000000 | 0.000000 | 0.037671 | 0 | 0 | 76 |
| badcal | original_test_all_10s+ | 0.808423 | 0.840659 | 0.822639 | 0.369830 | 514 | 492 | 68 |
| badcal | original_all_10s+ | 0.819092 | 0.731385 | 0.899887 | 0.939451 | 4430 | 734 | 127 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.113014 | 0.000000 | 0.000000 | 0.113014 | 0 | 0 | 68 |

## Error Counts

- test errors raw: 1500
- bad outlier errors raw: 281
- bad core errors raw: 4
- good->medium raw: 531
- medium->good raw: 635
- nonbad->bad raw: 49

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_shift_stress_pretrain/original_error_waveform_panels.png)
