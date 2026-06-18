# Original Candidate Error Audit: featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.863749 | 0.889560 | 0.864437 | 0.627737 | 397 | 424 | 48 |
| raw | original_all_10s+ | 0.867338 | 0.818401 | 0.900263 | 0.958940 | 3088 | 872 | 111 |
| raw | bad_core_nearboundary | 0.966387 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 |
| raw | bad_outlier_stress | 0.489726 | 0.000000 | 0.000000 | 0.489726 | 0 | 0 | 44 |
| badcal | original_test_all_10s+ | 0.864103 | 0.889560 | 0.864437 | 0.635036 | 397 | 424 | 48 |
| badcal | original_all_10s+ | 0.867429 | 0.818401 | 0.900263 | 0.959508 | 3088 | 872 | 111 |
| badcal | bad_core_nearboundary | 0.966387 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 |
| badcal | bad_outlier_stress | 0.500000 | 0.000000 | 0.000000 | 0.500000 | 0 | 0 | 44 |

## Error Counts

- test errors raw: 1155
- bad outlier errors raw: 149
- bad core errors raw: 4
- good->medium raw: 397
- medium->good raw: 424
- nonbad->bad raw: 181

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050/original_error_waveform_panels.png)
