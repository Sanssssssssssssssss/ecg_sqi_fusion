# Original Candidate Error Audit: predtop20_sqiquery_subject111_impulsebad_dual_p20_precisionselect

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.807125 | 0.873901 | 0.795075 | 0.345499 | 429 | 691 | 79 |
| raw | original_all_10s+ | 0.816392 | 0.740949 | 0.877117 | 0.937559 | 4353 | 957 | 139 |
| raw | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| raw | bad_outlier_stress | 0.085616 | 0.000000 | 0.000000 | 0.085616 | 0 | 0 | 77 |
| badcal | original_test_all_10s+ | 0.807007 | 0.873901 | 0.794849 | 0.345499 | 429 | 691 | 79 |
| badcal | original_all_10s+ | 0.816240 | 0.740949 | 0.876647 | 0.937559 | 4352 | 957 | 139 |
| badcal | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| badcal | bad_outlier_stress | 0.085616 | 0.000000 | 0.000000 | 0.085616 | 0 | 0 | 77 |

## Error Counts

- test errors raw: 1635
- bad outlier errors raw: 267
- bad core errors raw: 2
- good->medium raw: 429
- medium->good raw: 691
- nonbad->bad raw: 246

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_impulsebad_dual_p20_precisionselect/original_error_waveform_panels.png)
