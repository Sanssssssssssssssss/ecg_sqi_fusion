# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_v5badbranch_guard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.857969 | 0.889835 | 0.884320 | 0.291971 | 398 | 485 | 99 |
| raw | original_all_10s+ | 0.865973 | 0.818459 | 0.909108 | 0.932450 | 3090 | 933 | 163 |
| raw | bad_core_nearboundary | 0.924370 | 0.000000 | 0.000000 | 0.924370 | 0 | 0 | 9 |
| raw | bad_outlier_stress | 0.034247 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 90 |
| badcal | original_test_all_10s+ | 0.861272 | 0.889835 | 0.879575 | 0.411192 | 395 | 467 | 83 |
| badcal | original_all_10s+ | 0.866853 | 0.818459 | 0.906944 | 0.942289 | 3086 | 915 | 145 |
| badcal | bad_core_nearboundary | 0.974790 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 3 |
| badcal | bad_outlier_stress | 0.181507 | 0.000000 | 0.000000 | 0.181507 | 0 | 0 | 80 |

## Error Counts

- test errors raw: 1204
- bad outlier errors raw: 282
- bad core errors raw: 9
- good->medium raw: 398
- medium->good raw: 485
- nonbad->bad raw: 30

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_v5badbranch_guard_a050/original_error_waveform_panels.png)
