# Original Candidate Error Audit: predtop20_sqiquery_subject111_mediumonly_shift_impulse

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.775392 | 0.909066 | 0.714641 | 0.245742 | 330 | 1253 | 70 |
| raw | original_all_10s+ | 0.845309 | 0.834008 | 0.822168 | 0.928288 | 2827 | 1869 | 138 |
| raw | bad_core_nearboundary | 0.831933 | 0.000000 | 0.000000 | 0.831933 | 0 | 0 | 20 |
| raw | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 50 |
| badcal | original_test_all_10s+ | 0.767135 | 0.906868 | 0.693177 | 0.326034 | 320 | 1196 | 44 |
| badcal | original_all_10s+ | 0.842214 | 0.832600 | 0.810877 | 0.936235 | 2790 | 1807 | 103 |
| badcal | bad_core_nearboundary | 0.966387 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 |
| badcal | bad_outlier_stress | 0.065068 | 0.000000 | 0.000000 | 0.065068 | 0 | 0 | 40 |

## Error Counts

- test errors raw: 1904
- bad outlier errors raw: 290
- bad core errors raw: 20
- good->medium raw: 330
- medium->good raw: 1253
- nonbad->bad raw: 11

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_mediumonly_shift_impulse/original_error_waveform_panels.png)
