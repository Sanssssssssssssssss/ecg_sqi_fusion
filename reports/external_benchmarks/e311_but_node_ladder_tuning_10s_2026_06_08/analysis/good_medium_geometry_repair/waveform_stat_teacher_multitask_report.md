# Waveform Stat Teacher Multitask

The model receives only waveform-derived statistics at inference, while synthetic 47-column features are used as teacher targets during training.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| statteacher_top20_balanced | synthetic_test | 0.993849 | 0.992791 | 0.987448 | 0.998377 | 0.983402 | 6 | 2 | 0 | 4 |
| statteacher_top20_balanced | original_test_all_10s+ | 0.821045 | 0.686615 | 0.860440 | 0.842070 | 0.245742 | 508 | 698 | 0 | 55 |
| statteacher_top20_balanced | bad_core_nearboundary | 0.848739 | 0.306061 | 0.000000 | 0.000000 | 0.848739 | 0 | 0 | 0 | 18 |
| statteacher_top20_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 37 |
| statteacher_top20_medium | synthetic_test | 0.992824 | 0.991475 | 0.989540 | 0.996753 | 0.979253 | 5 | 4 | 0 | 5 |
| statteacher_top20_medium | original_test_all_10s+ | 0.820927 | 0.582426 | 0.841758 | 0.876638 | 0.036496 | 576 | 546 | 0 | 163 |
| statteacher_top20_medium | bad_core_nearboundary | 0.126050 | 0.074627 | 0.000000 | 0.000000 | 0.126050 | 0 | 0 | 0 | 104 |
| statteacher_top20_medium | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 59 |
| statteacher_geomcore_badguard | synthetic_test | 0.993337 | 0.992319 | 0.991632 | 0.995942 | 0.983402 | 4 | 5 | 0 | 4 |
| statteacher_geomcore_badguard | original_test_all_10s+ | 0.821871 | 0.678959 | 0.854396 | 0.850429 | 0.226277 | 530 | 661 | 0 | 72 |
| statteacher_geomcore_badguard | bad_core_nearboundary | 0.781513 | 0.292453 | 0.000000 | 0.000000 | 0.781513 | 0 | 0 | 0 | 26 |
| statteacher_geomcore_badguard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 46 |

## Teacher Recovery

| Candidate | Split | Mean MAE-z | Mean Corr |
|---|---|---:|---:|
| statteacher_geomcore_badguard | test | 0.6950 | 0.4296 |
| statteacher_geomcore_badguard | val | 0.6758 | 0.5052 |
| statteacher_top20_balanced | test | 0.6127 | 0.5286 |
| statteacher_top20_balanced | val | 0.5488 | 0.6301 |
| statteacher_top20_medium | test | 0.6121 | 0.5330 |
| statteacher_top20_medium | val | 0.5528 | 0.6290 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stat_teacher_multitask_metrics.csv`
- Teacher recovery CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stat_teacher_multitask_recovery.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stat_teacher_multitask_summary.json`
