# Original Candidate Error Audit: featurefirst_top20_hardrec_qrslite_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.823758 | 0.885440 | 0.827610 | 0.236010 | 417 | 758 | 99 |
| raw | original_all_10s+ | 0.853441 | 0.809130 | 0.888502 | 0.925828 | 3251 | 1175 | 175 |
| raw | bad_core_nearboundary | 0.747899 | 0.000000 | 0.000000 | 0.747899 | 0 | 0 | 30 |
| raw | bad_outlier_stress | 0.027397 | 0.000000 | 0.000000 | 0.027397 | 0 | 0 | 69 |
| badcal | original_test_all_10s+ | 0.822343 | 0.882418 | 0.817894 | 0.338200 | 414 | 724 | 63 |
| badcal | original_all_10s+ | 0.852227 | 0.807311 | 0.882386 | 0.936424 | 3241 | 1141 | 125 |
| badcal | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| badcal | bad_outlier_stress | 0.075342 | 0.000000 | 0.000000 | 0.075342 | 0 | 0 | 61 |

## Error Counts

- test errors raw: 1494
- bad outlier errors raw: 284
- bad core errors raw: 30
- good->medium raw: 417
- medium->good raw: 758
- nonbad->bad raw: 5

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_hardrec_qrslite_a050/original_error_waveform_panels.png)
