# Original Candidate Error Audit: predtop20_sqiquery_subject111_medium_gapheavy_balanced

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.771263 | 0.923352 | 0.695210 | 0.243309 | 279 | 1340 | 93 |
| raw | original_all_10s+ | 0.846189 | 0.904184 | 0.712458 | 0.928098 | 1632 | 3043 | 159 |
| raw | bad_core_nearboundary | 0.823529 | 0.000000 | 0.000000 | 0.823529 | 0 | 0 | 21 |
| raw | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 72 |
| badcal | original_test_all_10s+ | 0.762888 | 0.918407 | 0.671939 | 0.364964 | 270 | 1227 | 61 |
| badcal | original_all_10s+ | 0.841000 | 0.899842 | 0.697874 | 0.939073 | 1594 | 2918 | 121 |
| badcal | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| badcal | bad_outlier_stress | 0.113014 | 0.000000 | 0.000000 | 0.113014 | 0 | 0 | 59 |

## Error Counts

- test errors raw: 1939
- bad outlier errors raw: 290
- bad core errors raw: 21
- good->medium raw: 279
- medium->good raw: 1340
- nonbad->bad raw: 9

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_medium_gapheavy_balanced/original_error_waveform_panels.png)
