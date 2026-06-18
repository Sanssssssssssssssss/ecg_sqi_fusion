# Original Candidate Error Audit: predtop20_sqiquery_subject111_recordbad_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.782824 | 0.897527 | 0.733168 | 0.301703 | 367 | 1161 | 60 |
| raw | original_all_10s+ | 0.829288 | 0.780907 | 0.855194 | 0.933207 | 3718 | 1511 | 124 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.017123 | 0.000000 | 0.000000 | 0.017123 | 0 | 0 | 60 |
| badcal | original_test_all_10s+ | 0.778105 | 0.895879 | 0.722775 | 0.330900 | 366 | 1089 | 51 |
| badcal | original_all_10s+ | 0.825373 | 0.778560 | 0.844373 | 0.938127 | 3710 | 1429 | 101 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.058219 | 0.000000 | 0.000000 | 0.058219 | 0 | 0 | 51 |

## Error Counts

- test errors raw: 1841
- bad outlier errors raw: 287
- bad core errors raw: 0
- good->medium raw: 367
- medium->good raw: 1161
- nonbad->bad raw: 26

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_recordbad_pretrain/original_error_waveform_panels.png)
