# Original Candidate Error Audit: featurefirst_top20_qrsbase_n7110_cleangeom_encoderlite_mediumguard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.757697 | 0.935165 | 0.655219 | 0.289538 | 233 | 1505 | 36 |
| raw | original_all_10s+ | 0.852652 | 0.930939 | 0.687524 | 0.932261 | 1173 | 3293 | 98 |
| raw | bad_core_nearboundary | 0.966387 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 |
| raw | bad_outlier_stress | 0.013699 | 0.000000 | 0.000000 | 0.013699 | 0 | 0 | 32 |
| badcal | original_test_all_10s+ | 0.759113 | 0.935165 | 0.650700 | 0.367397 | 231 | 1489 | 28 |
| badcal | original_all_10s+ | 0.853046 | 0.930939 | 0.685171 | 0.939451 | 1171 | 3277 | 85 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.109589 | 0.000000 | 0.000000 | 0.109589 | 0 | 0 | 28 |

## Error Counts

- test errors raw: 2054
- bad outlier errors raw: 288
- bad core errors raw: 4
- good->medium raw: 233
- medium->good raw: 1505
- nonbad->bad raw: 24

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_n7110_cleangeom_encoderlite_mediumguard_a050/original_error_waveform_panels.png)
