# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_v5badbranch_recall_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.858087 | 0.889835 | 0.882512 | 0.313869 | 397 | 480 | 94 |
| raw | original_all_10s+ | 0.865973 | 0.818459 | 0.908261 | 0.934153 | 3089 | 928 | 158 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.054795 | 0.000000 | 0.000000 | 0.054795 | 0 | 0 | 88 |
| badcal | original_test_all_10s+ | 0.862215 | 0.889835 | 0.877316 | 0.454988 | 395 | 451 | 78 |
| badcal | original_all_10s+ | 0.867096 | 0.818459 | 0.906003 | 0.945695 | 3086 | 899 | 140 |
| badcal | bad_core_nearboundary | 0.974790 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 3 |
| badcal | bad_outlier_stress | 0.243151 | 0.000000 | 0.000000 | 0.243151 | 0 | 0 | 75 |

## Error Counts

- test errors raw: 1203
- bad outlier errors raw: 276
- bad core errors raw: 6
- good->medium raw: 397
- medium->good raw: 480
- nonbad->bad raw: 44

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_v5badbranch_recall_a050/original_error_waveform_panels.png)
