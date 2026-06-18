# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_flatbadaddon_balanced_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.858912 | 0.889835 | 0.883642 | 0.318735 | 399 | 479 | 94 |
| raw | original_all_10s+ | 0.866276 | 0.818459 | 0.908920 | 0.934721 | 3092 | 927 | 158 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.061644 | 0.000000 | 0.000000 | 0.061644 | 0 | 0 | 88 |
| badcal | original_test_all_10s+ | 0.863513 | 0.889835 | 0.876186 | 0.493917 | 395 | 449 | 67 |
| badcal | original_all_10s+ | 0.867399 | 0.818459 | 0.905344 | 0.948912 | 3086 | 897 | 128 |
| badcal | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| badcal | bad_outlier_stress | 0.291096 | 0.000000 | 0.000000 | 0.291096 | 0 | 0 | 66 |

## Error Counts

- test errors raw: 1196
- bad outlier errors raw: 274
- bad core errors raw: 6
- good->medium raw: 399
- medium->good raw: 479
- nonbad->bad raw: 38

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_flatbadaddon_balanced_a050/original_error_waveform_panels.png)
