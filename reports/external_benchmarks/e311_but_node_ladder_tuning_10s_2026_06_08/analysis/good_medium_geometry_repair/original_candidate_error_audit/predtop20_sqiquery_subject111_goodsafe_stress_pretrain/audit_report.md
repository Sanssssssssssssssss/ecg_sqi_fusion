# Original Candidate Error Audit: predtop20_sqiquery_subject111_goodsafe_stress_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.797688 | 0.913736 | 0.751920 | 0.262774 | 311 | 1089 | 64 |
| raw | original_all_10s+ | 0.842790 | 0.805903 | 0.859710 | 0.927720 | 3303 | 1473 | 141 |
| raw | bad_core_nearboundary | 0.865546 | 0.000000 | 0.000000 | 0.865546 | 0 | 0 | 16 |
| raw | bad_outlier_stress | 0.017123 | 0.000000 | 0.000000 | 0.017123 | 0 | 0 | 48 |
| badcal | original_test_all_10s+ | 0.795329 | 0.910440 | 0.743109 | 0.338200 | 310 | 1004 | 45 |
| badcal | original_all_10s+ | 0.840879 | 0.803028 | 0.853782 | 0.936991 | 3293 | 1383 | 104 |
| badcal | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| badcal | bad_outlier_stress | 0.075342 | 0.000000 | 0.000000 | 0.075342 | 0 | 0 | 43 |

## Error Counts

- test errors raw: 1715
- bad outlier errors raw: 287
- bad core errors raw: 16
- good->medium raw: 311
- medium->good raw: 1089
- nonbad->bad raw: 12

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_goodsafe_stress_pretrain/original_error_waveform_panels.png)
