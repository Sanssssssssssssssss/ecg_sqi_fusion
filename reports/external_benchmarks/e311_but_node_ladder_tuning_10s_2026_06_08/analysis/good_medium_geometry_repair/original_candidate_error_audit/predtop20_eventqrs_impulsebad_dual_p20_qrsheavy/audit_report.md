# Original Candidate Error Audit: predtop20_eventqrs_impulsebad_dual_p20_qrsheavy

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.815737 | 0.893956 | 0.808631 | 0.199513 | 386 | 841 | 111 |
| raw | original_all_10s+ | 0.861634 | 0.834243 | 0.874294 | 0.924503 | 2825 | 1329 | 180 |
| raw | bad_core_nearboundary | 0.663866 | 0.000000 | 0.000000 | 0.663866 | 0 | 0 | 40 |
| raw | bad_outlier_stress | 0.010274 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 71 |
| badcal | original_test_all_10s+ | 0.819394 | 0.893956 | 0.804112 | 0.323601 | 386 | 839 | 68 |
| badcal | original_all_10s+ | 0.862180 | 0.834243 | 0.869872 | 0.936802 | 2825 | 1327 | 123 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.047945 | 0.000000 | 0.000000 | 0.047945 | 0 | 0 | 68 |

## Error Counts

- test errors raw: 1562
- bad outlier errors raw: 289
- bad core errors raw: 40
- good->medium raw: 386
- medium->good raw: 841
- nonbad->bad raw: 6

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_eventqrs_impulsebad_dual_p20_qrsheavy/original_error_waveform_panels.png)
