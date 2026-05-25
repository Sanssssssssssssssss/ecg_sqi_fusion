# E3.10 Visual Transformer Tuning

Dataset: `e310_smooth_morph_mild_snr`.
Goal: improve the current single-model visual benchmark while keeping the same simple CLS raw transformer family.

## Baselines

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| D1 reference | 0.9465 | 0.9153 | 0.9465 | 0.9777 | 0.013/-0.710/-1.042 | `[[616, 53, 4], [22, 637, 14], [2, 13, 658]]` |
| E3.10 M0 baseline | 0.9271 | 0.9262 | 0.9020 | 0.9529 | 0.798/1.256/1.590 | `[[728, 57, 1], [50, 709, 27], [4, 33, 749]]` |
| E3.10 M1 D1 warm-start | 0.9334 | 0.9542 | 0.8906 | 0.9555 | 0.527/1.183/1.718 | `[[750, 32, 4], [66, 700, 20], [6, 29, 751]]` |
| E3.10 M2 warm-start + SNR head | 0.9402 | 0.9491 | 0.9135 | 0.9580 | 0.584/1.190/1.689 | `[[746, 35, 5], [48, 718, 20], [7, 26, 753]]` |
| E3.10 M3 low denoise | 0.9372 | 0.9389 | 0.9173 | 0.9555 | -0.491/0.508/1.105 | `[[738, 43, 5], [45, 721, 20], [7, 28, 751]]` |
| E3.10 M4 noise type | 0.9364 | 0.9427 | 0.9173 | 0.9491 | -0.680/0.292/1.029 | `[[741, 43, 2], [50, 721, 15], [7, 33, 746]]` |

## Round 1

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Denoise SNR Improve G/M/B | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| R1 M2 seed 1 | 0.9381 | 0.9377 | 0.9186 | 0.9580 | -7.745/-5.091/-4.041 | `[[737, 43, 6], [42, 722, 22], [7, 26, 753]]` |
| R1 M2 seed 2 | 0.9334 | 0.9326 | 0.9262 | 0.9415 | -7.338/-4.663/-3.481 | `[[733, 48, 5], [43, 728, 15], [11, 35, 740]]` |
| R1 M2 seed 3 | pending |  |  |  |  |  |
| R1 SNR lambda 0.02 | pending |  |  |  |  |  |
| R1 SNR lambda 0.075 | pending |  |  |  |  |  |
| R1 medium weight 1.03 | 0.9347 | 0.9389 | 0.9084 | 0.9567 | -8.408/-5.557/-4.521 | `[[738, 43, 5], [47, 714, 25], [8, 26, 752]]` |
| R1 medium weight 1.05 | pending |  |  |  |  |  |
| R1 label smoothing 0.005 | 0.9338 | 0.9440 | 0.9122 | 0.9453 | -7.824/-5.061/-3.977 | `[[742, 39, 5], [49, 717, 20], [10, 33, 743]]` |

Best E3.10 visual result: `E3.10 M2 warm-start + SNR head` = `0.9402`

## Interpretation

- M2 is the current single-model anchor because it already reaches the 0.94 visual benchmark threshold.
- Round 1 keeps the architecture fixed and only tests seed robustness, SNR-head weight, light label smoothing, and small medium-class weighting.
- If Round 1 does not improve M2, the next useful step is ensemble/error audit rather than adding heads.
