# Original Candidate Error Audit: predtop20_sqiquery_subject111_impulsebad_dual_p35

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.778223 | 0.904670 | 0.716900 | 0.318735 | 335 | 1142 | 51 |
| raw | original_all_10s+ | 0.843640 | 0.828551 | 0.821603 | 0.936613 | 2900 | 1676 | 105 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.041096 | 0.000000 | 0.000000 | 0.041096 | 0 | 0 | 51 |
| badcal | original_test_all_10s+ | 0.778223 | 0.904670 | 0.716900 | 0.318735 | 335 | 1142 | 51 |
| badcal | original_all_10s+ | 0.843640 | 0.828551 | 0.821603 | 0.936613 | 2900 | 1676 | 105 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.041096 | 0.000000 | 0.000000 | 0.041096 | 0 | 0 | 51 |

## Error Counts

- test errors raw: 1880
- bad outlier errors raw: 280
- bad core errors raw: 0
- good->medium raw: 335
- medium->good raw: 1142
- nonbad->bad raw: 123

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_impulsebad_dual_p35/original_error_waveform_panels.png)
