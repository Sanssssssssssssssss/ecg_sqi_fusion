# Original Candidate Error Audit: featurefirst_top20_rawbeat_artifact_auxctx_veto_finetune_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.826000 | 0.800824 | 0.869408 | 0.581509 | 705 | 396 | 81 |
| raw | original_all_10s+ | 0.750303 | 0.571144 | 0.934795 | 0.957048 | 7266 | 484 | 135 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.410959 | 0.000000 | 0.000000 | 0.410959 | 0 | 0 | 81 |
| badcal | original_test_all_10s+ | 0.825882 | 0.800824 | 0.869182 | 0.581509 | 705 | 396 | 81 |
| badcal | original_all_10s+ | 0.750273 | 0.571144 | 0.934701 | 0.957048 | 7266 | 484 | 135 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.410959 | 0.000000 | 0.000000 | 0.410959 | 0 | 0 | 81 |

## Error Counts

- test errors raw: 1475
- bad outlier errors raw: 172
- bad core errors raw: 0
- good->medium raw: 705
- medium->good raw: 396
- nonbad->bad raw: 202

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_rawbeat_artifact_auxctx_veto_finetune_a050/original_error_waveform_panels.png)
