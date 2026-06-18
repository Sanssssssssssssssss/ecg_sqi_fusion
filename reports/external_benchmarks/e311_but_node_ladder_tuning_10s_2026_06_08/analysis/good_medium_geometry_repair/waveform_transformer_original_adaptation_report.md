# Waveform-Only Transformer Original Adaptation

Diagnostic only: original BUT train labels are used for training, val for selection, test held out. Inputs remain waveform only.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| orig_convtx_robust3_aux | original_val | 0.983578 | 0.964584 | 0.991744 | 0.904762 | 0.987952 | 8 | 10 | 1 |
| orig_convtx_robust3_aux | original_test_all_10s+ | 0.762062 | 0.660704 | 0.969231 | 0.635337 | 0.291971 | 112 | 1600 | 162 |
| orig_convtx_robust3_aux_badcal | original_test_all_10s+ | 0.745193 | 0.655685 | 0.969231 | 0.587438 | 0.459854 | 111 | 1595 | 95 |
| orig_convtx_robust3_aux | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| orig_convtx_robust3_aux_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| orig_convtx_robust3_aux | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 162 |
| orig_convtx_robust3_aux_badcal | bad_outlier_stress | 0.239726 | 0.128913 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 95 |

## Candidates

- `orig_convtx_robust3_aux`: best_epoch=4, threshold=0.03, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_transformer_original_adaptation\N17043_gm_probe\orig_convtx_robust3_aux\ckpt_best.pt`

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_original_adaptation_metrics.csv`
