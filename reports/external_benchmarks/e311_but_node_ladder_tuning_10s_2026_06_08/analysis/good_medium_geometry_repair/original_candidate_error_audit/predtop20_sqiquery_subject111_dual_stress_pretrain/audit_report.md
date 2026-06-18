# Original Candidate Error Audit: predtop20_sqiquery_subject111_dual_stress_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.811136 | 0.891209 | 0.793945 | 0.287105 | 385 | 856 | 71 |
| raw | original_all_10s+ | 0.847312 | 0.809306 | 0.866391 | 0.931504 | 3229 | 1353 | 135 |
| raw | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| raw | bad_outlier_stress | 0.013699 | 0.000000 | 0.000000 | 0.013699 | 0 | 0 | 66 |
| badcal | original_test_all_10s+ | 0.802524 | 0.884615 | 0.779485 | 0.323601 | 382 | 841 | 61 |
| badcal | original_all_10s+ | 0.844156 | 0.807076 | 0.858205 | 0.935478 | 3220 | 1335 | 122 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.047945 | 0.000000 | 0.000000 | 0.047945 | 0 | 0 | 61 |

## Error Counts

- test errors raw: 1601
- bad outlier errors raw: 288
- bad core errors raw: 5
- good->medium raw: 385
- medium->good raw: 856
- nonbad->bad raw: 67

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_dual_stress_pretrain/original_error_waveform_panels.png)
