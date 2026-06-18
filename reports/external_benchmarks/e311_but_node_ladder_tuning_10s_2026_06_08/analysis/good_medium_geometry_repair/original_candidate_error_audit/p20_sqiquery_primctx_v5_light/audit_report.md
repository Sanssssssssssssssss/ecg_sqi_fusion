# Original Candidate Error Audit: p20_sqiquery_primctx_v5_light

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.808069 | 0.894231 | 0.783552 | 0.309002 | 375 | 897 | 68 |
| raw | original_all_10s+ | 0.850983 | 0.820161 | 0.858769 | 0.934721 | 3043 | 1423 | 128 |
| raw | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| raw | bad_outlier_stress | 0.034247 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 66 |
| badcal | original_test_all_10s+ | 0.803940 | 0.889835 | 0.777000 | 0.333333 | 375 | 875 | 64 |
| badcal | original_all_10s+ | 0.848556 | 0.818635 | 0.852465 | 0.937181 | 3041 | 1399 | 121 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.061644 | 0.000000 | 0.000000 | 0.061644 | 0 | 0 | 64 |

## Error Counts

- test errors raw: 1627
- bad outlier errors raw: 282
- bad core errors raw: 2
- good->medium raw: 375
- medium->good raw: 897
- nonbad->bad raw: 71

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/p20_sqiquery_primctx_v5_light/original_error_waveform_panels.png)
