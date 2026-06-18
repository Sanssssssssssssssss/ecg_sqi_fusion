# Original Candidate Error Audit: predtop20_sqiquery_subject111_recordbad_dual_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.773151 | 0.899176 | 0.738138 | 0.034063 | 332 | 830 | 169 |
| raw | original_all_10s+ | 0.677054 | 0.774277 | 0.856511 | 0.002649 | 3784 | 1169 | 5041 |
| raw | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 |
| raw | bad_outlier_stress | 0.047945 | 0.000000 | 0.000000 | 0.047945 | 0 | 0 | 50 |
| badcal | original_test_all_10s+ | 0.737761 | 0.857418 | 0.701085 | 0.072993 | 313 | 589 | 169 |
| badcal | original_all_10s+ | 0.664522 | 0.759608 | 0.839292 | 0.006433 | 3757 | 916 | 5037 |
| badcal | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 |
| badcal | bad_outlier_stress | 0.102740 | 0.000000 | 0.000000 | 0.102740 | 0 | 0 | 50 |

## Error Counts

- test errors raw: 1923
- bad outlier errors raw: 278
- bad core errors raw: 119
- good->medium raw: 332
- medium->good raw: 830
- nonbad->bad raw: 364

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_recordbad_dual_pretrain/original_error_waveform_panels.png)
