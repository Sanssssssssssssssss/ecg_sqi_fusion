# Original Candidate Error Audit: featurefirst_top20_hardrec_eventqrs_gate_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.813967 | 0.884890 | 0.803886 | 0.294404 | 411 | 843 | 96 |
| raw | original_all_10s+ | 0.832595 | 0.765534 | 0.890290 | 0.932829 | 3985 | 1133 | 159 |
| raw | bad_core_nearboundary | 0.974790 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 3 |
| raw | bad_outlier_stress | 0.017123 | 0.000000 | 0.000000 | 0.017123 | 0 | 0 | 93 |
| badcal | original_test_all_10s+ | 0.810074 | 0.882692 | 0.795300 | 0.326034 | 410 | 788 | 88 |
| badcal | original_all_10s+ | 0.830805 | 0.764185 | 0.885209 | 0.936235 | 3974 | 1077 | 146 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.051370 | 0.000000 | 0.000000 | 0.051370 | 0 | 0 | 88 |

## Error Counts

- test errors raw: 1577
- bad outlier errors raw: 287
- bad core errors raw: 3
- good->medium raw: 411
- medium->good raw: 843
- nonbad->bad raw: 33

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_hardrec_eventqrs_gate_a050/original_error_waveform_panels.png)
