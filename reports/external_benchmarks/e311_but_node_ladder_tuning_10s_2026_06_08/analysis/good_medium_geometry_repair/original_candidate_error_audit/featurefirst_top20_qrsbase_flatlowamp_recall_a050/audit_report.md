# Original Candidate Error Audit: featurefirst_top20_qrsbase_flatlowamp_recall_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.856317 | 0.889835 | 0.882964 | 0.272506 | 400 | 468 | 122 |
| raw | original_all_10s+ | 0.865518 | 0.818459 | 0.908826 | 0.930180 | 3093 | 916 | 191 |
| raw | bad_core_nearboundary | 0.647059 | 0.000000 | 0.000000 | 0.647059 | 0 | 0 | 42 |
| raw | bad_outlier_stress | 0.119863 | 0.000000 | 0.000000 | 0.119863 | 0 | 0 | 80 |
| badcal | original_test_all_10s+ | 0.861744 | 0.889835 | 0.873701 | 0.484185 | 395 | 440 | 67 |
| badcal | original_all_10s+ | 0.867004 | 0.818459 | 0.904309 | 0.948534 | 3086 | 888 | 126 |
| badcal | bad_core_nearboundary | 0.966387 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 |
| badcal | bad_outlier_stress | 0.287671 | 0.000000 | 0.000000 | 0.287671 | 0 | 0 | 63 |

## Error Counts

- test errors raw: 1218
- bad outlier errors raw: 257
- bad core errors raw: 42
- good->medium raw: 400
- medium->good raw: 468
- nonbad->bad raw: 51

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_flatlowamp_recall_a050/original_error_waveform_panels.png)
