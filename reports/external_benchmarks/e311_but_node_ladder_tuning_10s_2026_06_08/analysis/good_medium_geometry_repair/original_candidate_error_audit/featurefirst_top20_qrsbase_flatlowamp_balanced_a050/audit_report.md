# Original Candidate Error Audit: featurefirst_top20_qrsbase_flatlowamp_balanced_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.856081 | 0.889835 | 0.886353 | 0.231144 | 400 | 481 | 129 |
| raw | original_all_10s+ | 0.865578 | 0.818459 | 0.910519 | 0.927152 | 3093 | 929 | 197 |
| raw | bad_core_nearboundary | 0.647059 | 0.000000 | 0.000000 | 0.647059 | 0 | 0 | 42 |
| raw | bad_outlier_stress | 0.061644 | 0.000000 | 0.000000 | 0.061644 | 0 | 0 | 87 |
| badcal | original_test_all_10s+ | 0.859856 | 0.889835 | 0.872797 | 0.454988 | 395 | 451 | 69 |
| badcal | original_all_10s+ | 0.866458 | 0.818459 | 0.903651 | 0.946452 | 3082 | 899 | 127 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.232877 | 0.000000 | 0.000000 | 0.232877 | 0 | 0 | 69 |

## Error Counts

- test errors raw: 1220
- bad outlier errors raw: 274
- bad core errors raw: 42
- good->medium raw: 400
- medium->good raw: 481
- nonbad->bad raw: 23

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_flatlowamp_balanced_a050/original_error_waveform_panels.png)
