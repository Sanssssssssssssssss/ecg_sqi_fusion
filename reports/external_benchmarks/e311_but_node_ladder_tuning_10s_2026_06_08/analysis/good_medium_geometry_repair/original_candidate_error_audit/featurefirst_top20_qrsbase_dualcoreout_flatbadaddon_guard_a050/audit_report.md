# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_flatbadaddon_guard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.858323 | 0.889835 | 0.883642 | 0.306569 | 399 | 484 | 95 |
| raw | original_all_10s+ | 0.866094 | 0.818459 | 0.908920 | 0.933586 | 3091 | 932 | 159 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.044521 | 0.000000 | 0.000000 | 0.044521 | 0 | 0 | 89 |
| badcal | original_test_all_10s+ | 0.863277 | 0.889835 | 0.876412 | 0.486618 | 395 | 454 | 67 |
| badcal | original_all_10s+ | 0.867308 | 0.818459 | 0.905438 | 0.948155 | 3086 | 902 | 129 |
| badcal | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| badcal | bad_outlier_stress | 0.280822 | 0.000000 | 0.000000 | 0.280822 | 0 | 0 | 66 |

## Error Counts

- test errors raw: 1201
- bad outlier errors raw: 279
- bad core errors raw: 6
- good->medium raw: 399
- medium->good raw: 484
- nonbad->bad raw: 33

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_flatbadaddon_guard_a050/original_error_waveform_panels.png)
