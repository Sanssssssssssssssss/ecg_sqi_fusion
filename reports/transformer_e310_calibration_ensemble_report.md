# E3.10 Calibration And Ensemble

Purpose: diagnose whether the current E3.10 visual benchmark is limited by model variance or by the data boundary.
The logit offsets are fit on validation only and then applied once to test.

Rows: val=2307, test=2358.
Best raw: `ensemble M2 seeds` = `0.9411`.
Best calibrated/ensemble: `ensemble M2 seeds` = `0.9419`.

| Run | Raw Test Acc | Calibrated Test Acc | Good Recall | Medium Recall | Bad Recall | Offsets [g,m,b] | Calibrated Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| M2 warm-start + SNR head | 0.9402 | 0.9402 | 0.9466 | 0.9160 | 0.9580 | `[-0.21, 0.04, 0.0]` | `[[744, 37, 5], [46, 720, 20], [7, 26, 753]]` |
| M3 low denoise | 0.9372 | 0.9377 | 0.9402 | 0.9173 | 0.9555 | `[0.06, 0.01, 0.0]` | `[[739, 42, 5], [46, 721, 19], [7, 28, 751]]` |
| R1 M2 seed 1 | 0.9381 | 0.9381 | 0.9427 | 0.9109 | 0.9606 | `[0.29, -0.24, 0.0]` | `[[741, 39, 6], [46, 716, 24], [8, 23, 755]]` |
| R1 M2 seed 2 | 0.9334 | 0.9351 | 0.9326 | 0.9262 | 0.9466 | `[-0.21, -0.29, 0.0]` | `[[733, 48, 5], [43, 728, 15], [11, 31, 744]]` |
| R1 M2 seed 3 | 0.9372 | 0.9372 | 0.9415 | 0.9198 | 0.9504 | `[0.0, 0.2, 0.0]` | `[[740, 42, 4], [46, 723, 17], [6, 33, 747]]` |
| R1 SNR lambda 0.02 | 0.9338 | 0.9338 | 0.9440 | 0.9122 | 0.9453 | `[0.03, 0.04, 0.0]` | `[[742, 42, 2], [50, 717, 19], [10, 33, 743]]` |
| R1 SNR lambda 0.075 | 0.9347 | 0.9351 | 0.9427 | 0.9097 | 0.9529 | `[0.0, 0.01, 0.0]` | `[[741, 40, 5], [49, 715, 22], [10, 27, 749]]` |
| R1 medium weight 1.03 | 0.9347 | 0.9355 | 0.9389 | 0.9097 | 0.9580 | `[-0.21, -0.2, 0.0]` | `[[738, 43, 5], [46, 715, 25], [8, 25, 753]]` |
| R1 medium weight 1.05 | 0.9321 | 0.9321 | 0.9402 | 0.9109 | 0.9453 | `[0.0, 0.09, 0.0]` | `[[739, 45, 2], [51, 716, 19], [10, 33, 743]]` |
| R1 label smoothing 0.005 | 0.9338 | 0.9313 | 0.9478 | 0.9008 | 0.9453 | `[0.32, -0.01, 0.0]` | `[[745, 37, 4], [59, 708, 19], [10, 33, 743]]` |
| ensemble M2 seeds | 0.9411 | 0.9419 | 0.9504 | 0.9186 | 0.9567 | `[0.03, -0.06, 0.0]` | `[[747, 34, 5], [46, 722, 18], [7, 27, 752]]` |
| ensemble SNR family | 0.9394 | 0.9411 | 0.9504 | 0.9173 | 0.9555 | `[0.24, 0.11, 0.0]` | `[[747, 34, 5], [46, 721, 19], [9, 26, 751]]` |
| ensemble denoise family | 0.9389 | 0.9385 | 0.9389 | 0.9211 | 0.9555 | `[-0.19, 0.18, 0.0]` | `[[738, 42, 6], [44, 724, 18], [7, 28, 751]]` |
| ensemble top5 mixed | 0.9411 | 0.9398 | 0.9466 | 0.9122 | 0.9606 | `[-0.32, -0.3, 0.0]` | `[[744, 36, 6], [45, 717, 24], [7, 24, 755]]` |
| ensemble all R1 selected | 0.9398 | 0.9377 | 0.9440 | 0.9211 | 0.9478 | `[0.27, 0.34, 0.0]` | `[[742, 40, 4], [46, 724, 16], [9, 32, 745]]` |

## Interpretation

- M2 remains the single-model anchor unless calibration or ensemble beats it on test.
- A useful ensemble gain would mean model variance still matters; no gain means the remaining errors are mostly label/data boundary.
- This is a diagnostic pass only: it does not change model structure, data generation, or training code.
