# Original Candidate Error Audit: featurefirst_top20_qrsbase_primres_current_balanced_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.825764 | 0.908516 | 0.789200 | 0.486618 | 327 | 785 | 45 |
| raw | original_all_10s+ | 0.866883 | 0.849968 | 0.853688 | 0.947966 | 2547 | 1389 | 108 |
| raw | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| raw | bad_outlier_stress | 0.294521 | 0.000000 | 0.000000 | 0.294521 | 0 | 0 | 40 |
| badcal | original_test_all_10s+ | 0.825764 | 0.908516 | 0.789200 | 0.486618 | 327 | 785 | 45 |
| badcal | original_all_10s+ | 0.866883 | 0.849968 | 0.853688 | 0.947966 | 2547 | 1389 | 108 |
| badcal | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| badcal | bad_outlier_stress | 0.294521 | 0.000000 | 0.000000 | 0.294521 | 0 | 0 | 40 |

## Error Counts

- test errors raw: 1477
- bad outlier errors raw: 206
- bad core errors raw: 5
- good->medium raw: 327
- medium->good raw: 785
- nonbad->bad raw: 154

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_primres_current_balanced_a050/original_error_waveform_panels.png)
