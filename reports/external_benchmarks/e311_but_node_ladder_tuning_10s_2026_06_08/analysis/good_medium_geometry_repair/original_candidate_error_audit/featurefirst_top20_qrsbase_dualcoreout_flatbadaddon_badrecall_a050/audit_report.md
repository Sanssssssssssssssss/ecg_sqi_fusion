# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_flatbadaddon_badrecall_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.858440 | 0.889835 | 0.882738 | 0.318735 | 398 | 480 | 93 |
| raw | original_all_10s+ | 0.866125 | 0.818459 | 0.908449 | 0.934721 | 3090 | 928 | 157 |
| raw | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| raw | bad_outlier_stress | 0.058219 | 0.000000 | 0.000000 | 0.058219 | 0 | 0 | 88 |
| badcal | original_test_all_10s+ | 0.862687 | 0.889560 | 0.874379 | 0.498783 | 395 | 450 | 68 |
| badcal | original_all_10s+ | 0.867126 | 0.818401 | 0.904498 | 0.949101 | 3086 | 898 | 130 |
| badcal | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| badcal | bad_outlier_stress | 0.297945 | 0.000000 | 0.000000 | 0.297945 | 0 | 0 | 67 |

## Error Counts

- test errors raw: 1200
- bad outlier errors raw: 275
- bad core errors raw: 5
- good->medium raw: 398
- medium->good raw: 480
- nonbad->bad raw: 42

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_flatbadaddon_badrecall_a050/original_error_waveform_panels.png)
