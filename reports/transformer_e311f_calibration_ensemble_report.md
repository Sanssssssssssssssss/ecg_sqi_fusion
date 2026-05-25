# E3.11f Calibration And Ensemble

Purpose: diagnose whether the remaining E3.11f error is a class-threshold issue or a model-variance issue.
The logit offsets are fit on validation only and then applied once to test.

Rows: val=2292, test=2325.
Best raw: `ensemble all selected` = `0.9402`.
Best calibrated/ensemble: `ensemble all selected` = `0.9398`.

| Run | Raw Test Acc | Calibrated Test Acc | Good Recall | Medium Recall | Bad Recall | Offsets [g,m,b] | Calibrated Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| R5 tiny denoise curriculum | 0.9381 | 0.9381 | 0.9277 | 0.9277 | 0.9587 | `[-0.16, -0.15, 0.0]` | `[[719, 53, 3], [51, 719, 5], [11, 21, 743]]` |
| R5 robust input | 0.9376 | 0.9372 | 0.9445 | 0.9071 | 0.9600 | `[0.25, 0.06, 0.0]` | `[[732, 41, 2], [60, 703, 12], [5, 26, 744]]` |
| R5 long early-stop | 0.9376 | 0.9368 | 0.9290 | 0.9174 | 0.9639 | `[0.0, -0.08, 0.0]` | `[[720, 51, 4], [52, 711, 12], [8, 20, 747]]` |
| R1 cls-only SNR 0.05 | 0.9376 | 0.9376 | 0.9277 | 0.9265 | 0.9587 | `[-0.03, 0.0, 0.0]` | `[[719, 54, 2], [52, 718, 5], [11, 21, 743]]` |
| R3 lr 4e-5 | 0.9376 | 0.9363 | 0.9252 | 0.9226 | 0.9613 | `[-0.17, -0.08, 0.0]` | `[[717, 55, 3], [52, 715, 8], [8, 22, 745]]` |
| R6 robust medium weight 1.03 | 0.9368 | 0.9368 | 0.9381 | 0.9135 | 0.9587 | `[0.0, -0.02, 0.0]` | `[[727, 45, 3], [54, 708, 13], [5, 27, 743]]` |
| R6 raw medium weight 1.03 | 0.9355 | 0.9351 | 0.9342 | 0.9123 | 0.9587 | `[-0.3, -0.23, 0.0]` | `[[724, 47, 4], [59, 707, 9], [10, 22, 743]]` |
| R4 seed 3 best recipe | 0.9368 | 0.9372 | 0.9226 | 0.9265 | 0.9626 | `[-0.02, 0.11, 0.0]` | `[[715, 58, 2], [44, 718, 13], [8, 21, 746]]` |
| R4 good weight 1.08 | 0.9363 | 0.9363 | 0.9316 | 0.9161 | 0.9613 | `[0.0, -0.01, 0.0]` | `[[722, 50, 3], [57, 710, 8], [9, 21, 745]]` |
| ensemble top3 mixed | 0.9389 | 0.9394 | 0.9290 | 0.9277 | 0.9613 | `[0.04, 0.23, 0.0]` | `[[720, 52, 3], [49, 719, 7], [9, 21, 745]]` |
| ensemble top5 mixed | 0.9385 | 0.9385 | 0.9329 | 0.9213 | 0.9613 | `[0.14, 0.03, 0.0]` | `[[723, 49, 3], [54, 714, 7], [10, 20, 745]]` |
| ensemble robust family | 0.9389 | 0.9389 | 0.9277 | 0.9277 | 0.9613 | `[-0.26, 0.27, 0.0]` | `[[719, 54, 2], [45, 719, 11], [3, 27, 745]]` |
| ensemble all selected | 0.9402 | 0.9398 | 0.9303 | 0.9277 | 0.9613 | `[0.0, 0.11, 0.0]` | `[[721, 51, 3], [49, 719, 7], [8, 22, 745]]` |

## Interpretation

- Calibration is only useful if calibrated test accuracy exceeds the raw best without harming one class badly.
- Ensemble is diagnostic: a large ensemble gain means model variance; little gain means the remaining errors are mostly data/label boundary.
- This script does not change model structure or training code.
