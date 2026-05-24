# E3.10/E3.11 SNR Warm-Start Sweep

Goal: test whether D1 warm-start plus explicit SNR learning can rescue the visual-SNR datasets without changing the model architecture.

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- |
| D1 E3.9a reference | 0.9465 | 0.9153 | 0.9465 | 0.9777 | `[[616, 53, 4], [22, 637, 14], [2, 13, 658]]` |
| E3.10 M0 baseline | 0.9271 | 0.9262 | 0.9020 | 0.9529 | `[[728, 57, 1], [50, 709, 27], [4, 33, 749]]` |
| E3.10 M1 D1 warm-start | missing |  |  |  | `/home/cx272/final_project/ecg_sqi_fusion/outputs/transformer_e310_smooth_morph_mild_snr/models/e310_m1_d1warm_lr3e5/test_report.json` |
| E3.10 M2 warm-start + SNR head | missing |  |  |  | `/home/cx272/final_project/ecg_sqi_fusion/outputs/transformer_e310_smooth_morph_mild_snr/models/e310_m2_d1warm_snr005/test_report.json` |
| E3.10 M3 warm-start + SNR head + low denoise | missing |  |  |  | `/home/cx272/final_project/ecg_sqi_fusion/outputs/transformer_e310_smooth_morph_mild_snr/models/e310_m3_d1warm_snr005_lowden/test_report.json` |
| E3.10 M4 M3 + noise type head | missing |  |  |  | `/home/cx272/final_project/ecg_sqi_fusion/outputs/transformer_e310_smooth_morph_mild_snr/models/e310_m4_d1warm_snr005_lowden_ntype005/test_report.json` |
| E3.11 M0 baseline | 0.8810 | 0.9004 | 0.8072 | 0.9353 | `[[696, 63, 14], [104, 624, 45], [9, 41, 723]]` |
| E3.11 previous best tune | 0.8698 | 0.8409 | 0.8292 | 0.9392 | `[[650, 113, 10], [75, 641, 57], [3, 44, 726]]` |
| E3.11 M1 D1 warm-start | missing |  |  |  | `/home/cx272/final_project/ecg_sqi_fusion/outputs/transformer_e311_visual_gap/models/e311_m1_d1warm_lr3e5/test_report.json` |
| E3.11 M2 warm-start + SNR head | missing |  |  |  | `/home/cx272/final_project/ecg_sqi_fusion/outputs/transformer_e311_visual_gap/models/e311_m2_d1warm_snr005/test_report.json` |
| E3.11 M3 warm-start + SNR head + low denoise | missing |  |  |  | `/home/cx272/final_project/ecg_sqi_fusion/outputs/transformer_e311_visual_gap/models/e311_m3_d1warm_snr005_lowden/test_report.json` |
| E3.11 M4 M3 + noise type head | missing |  |  |  | `/home/cx272/final_project/ecg_sqi_fusion/outputs/transformer_e311_visual_gap/models/e311_m4_d1warm_snr005_lowden_ntype005/test_report.json` |

## Best New Runs

- E3.10 best new run: pending
- E3.11 best new run: pending

Success criteria:

- E3.11 `>=0.90`: rescued enough for further analysis
- E3.11 `>=0.93`: usable visual benchmark
- E3.11 `>=0.94`: strong visual benchmark
- E3.10 `>=0.94`: candidate visual version if E3.11 remains too severe
