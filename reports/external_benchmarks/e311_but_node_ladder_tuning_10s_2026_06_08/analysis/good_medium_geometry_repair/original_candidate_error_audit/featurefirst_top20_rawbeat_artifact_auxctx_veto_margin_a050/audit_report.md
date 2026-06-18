# Original Candidate Error Audit: featurefirst_top20_rawbeat_artifact_auxctx_veto_margin_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.795211 | 0.867582 | 0.769544 | 0.430657 | 469 | 887 | 59 |
| raw | original_all_10s+ | 0.791844 | 0.687438 | 0.883233 | 0.944749 | 5293 | 1083 | 116 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.198630 | 0.000000 | 0.000000 | 0.198630 | 0 | 0 | 59 |
| badcal | original_test_all_10s+ | 0.799693 | 0.865385 | 0.763669 | 0.605839 | 469 | 772 | 47 |
| badcal | original_all_10s+ | 0.790357 | 0.683800 | 0.877305 | 0.959130 | 5285 | 955 | 100 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.445205 | 0.000000 | 0.000000 | 0.445205 | 0 | 0 | 47 |

## Error Counts

- test errors raw: 1736
- bad outlier errors raw: 234
- bad core errors raw: 0
- good->medium raw: 469
- medium->good raw: 887
- nonbad->bad raw: 146

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_rawbeat_artifact_auxctx_veto_margin_a050/original_error_waveform_panels.png)
