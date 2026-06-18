# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_v5badbranch_balanced_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.857851 | 0.889835 | 0.884546 | 0.287105 | 398 | 486 | 100 |
| raw | original_all_10s+ | 0.865942 | 0.818459 | 0.909202 | 0.932072 | 3090 | 934 | 164 |
| raw | bad_core_nearboundary | 0.941176 | 0.000000 | 0.000000 | 0.941176 | 0 | 0 | 7 |
| raw | bad_outlier_stress | 0.020548 | 0.000000 | 0.000000 | 0.020548 | 0 | 0 | 93 |
| badcal | original_test_all_10s+ | 0.859738 | 0.889835 | 0.879575 | 0.379562 | 395 | 468 | 84 |
| badcal | original_all_10s+ | 0.866458 | 0.818459 | 0.906944 | 0.939830 | 3086 | 916 | 146 |
| badcal | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| badcal | bad_outlier_stress | 0.133562 | 0.000000 | 0.000000 | 0.133562 | 0 | 0 | 82 |

## Error Counts

- test errors raw: 1205
- bad outlier errors raw: 286
- bad core errors raw: 7
- good->medium raw: 398
- medium->good raw: 486
- nonbad->bad raw: 28

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_v5badbranch_balanced_a050/original_error_waveform_panels.png)
