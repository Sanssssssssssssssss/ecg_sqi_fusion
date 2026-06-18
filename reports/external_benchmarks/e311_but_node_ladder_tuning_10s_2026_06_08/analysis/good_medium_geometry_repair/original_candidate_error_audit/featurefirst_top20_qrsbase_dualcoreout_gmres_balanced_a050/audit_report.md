# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_gmres_balanced_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.858794 | 0.892582 | 0.854270 | 0.608273 | 386 | 450 | 49 |
| raw | original_all_10s+ | 0.867854 | 0.824033 | 0.893677 | 0.957237 | 2993 | 925 | 113 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.469178 | 0.000000 | 0.000000 | 0.469178 | 0 | 0 | 43 |
| badcal | original_test_all_10s+ | 0.858794 | 0.892582 | 0.854270 | 0.608273 | 386 | 450 | 49 |
| badcal | original_all_10s+ | 0.867854 | 0.824033 | 0.893677 | 0.957237 | 2993 | 925 | 113 |
| badcal | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| badcal | bad_outlier_stress | 0.469178 | 0.000000 | 0.000000 | 0.469178 | 0 | 0 | 43 |

## Error Counts

- test errors raw: 1197
- bad outlier errors raw: 155
- bad core errors raw: 6
- good->medium raw: 386
- medium->good raw: 450
- nonbad->bad raw: 200

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_gmres_balanced_a050/original_error_waveform_panels.png)
