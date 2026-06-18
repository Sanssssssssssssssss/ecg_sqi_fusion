# Original Candidate Error Audit: featuregate_top20_shift_stress_a050_bneg

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.809131 | 0.904945 | 0.783552 | 0.236010 | 345 | 939 | 72 |
| raw | original_all_10s+ | 0.855808 | 0.830018 | 0.861122 | 0.928288 | 2896 | 1451 | 135 |
| raw | bad_core_nearboundary | 0.781513 | 0.000000 | 0.000000 | 0.781513 | 0 | 0 | 26 |
| raw | bad_outlier_stress | 0.013699 | 0.000000 | 0.000000 | 0.013699 | 0 | 0 | 46 |
| badcal | original_test_all_10s+ | 0.805710 | 0.901923 | 0.770673 | 0.330900 | 343 | 908 | 43 |
| badcal | original_all_10s+ | 0.854200 | 0.828903 | 0.853782 | 0.936613 | 2891 | 1420 | 101 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.058219 | 0.000000 | 0.000000 | 0.058219 | 0 | 0 | 43 |

## Error Counts

- test errors raw: 1618
- bad outlier errors raw: 288
- bad core errors raw: 26
- good->medium raw: 345
- medium->good raw: 939
- nonbad->bad raw: 20

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featuregate_top20_shift_stress_a050_bneg/original_error_waveform_panels.png)
