# E3.11f Logit Calibration

Validation-only class logit offsets are tuned on val and applied once to test.
This does not retrain the model or change the dataset.

| Run | Raw Test Acc | Cal Val Acc | Cal Test Acc | Good Recall | Medium Recall | Bad Recall | Offset [G,M,B] | Calibrated CM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| e311f_lite_e310_morph_r2_r1_cls_only_snr005_ls003 | 0.9372 | 0.9394 | 0.9376 | 0.9277 | 0.9265 | 0.9587 | `[0.0975, 0.2975, 0.0]` | `[[719, 53, 3], [47, 718, 10], [8, 24, 743]]` |
| ensemble_r1_005_plus_ls003 | 0.9359 | 0.9407 | 0.9359 | 0.9174 | 0.9303 | 0.9600 | `[-0.1875, 0.2175, 0.0]` | `[[711, 60, 4], [47, 721, 7], [7, 24, 744]]` |
| ensemble_r1_005_010_ls003 | 0.9381 | 0.9415 | 0.9359 | 0.9148 | 0.9316 | 0.9613 | `[-0.405, 0.2575, 0.0]` | `[[709, 62, 4], [44, 722, 9], [7, 23, 745]]` |
| ensemble_all_four | 0.9376 | 0.9407 | 0.9359 | 0.9135 | 0.9355 | 0.9587 | `[-0.15, 0.4825, 0.0]` | `[[708, 63, 4], [42, 725, 8], [7, 25, 743]]` |
| e311f_lite_e310_morph_r1_cls_only_snr005 | 0.9376 | 0.9385 | 0.9342 | 0.9406 | 0.9006 | 0.9613 | `[0.53, -0.4425, 0.0]` | `[[729, 43, 3], [71, 698, 6], [11, 19, 745]]` |
| e311f_lite_e310_morph_r1_cls_only_snr010 | 0.9355 | 0.9372 | 0.9325 | 0.9123 | 0.9252 | 0.9600 | `[0.0, 0.5475, 0.0]` | `[[707, 64, 4], [49, 717, 9], [7, 24, 744]]` |
| e311f_lite_e310_morph_r2_r1_cls_only_snr005_lr2e5 | 0.9303 | 0.9411 | 0.9299 | 0.9174 | 0.9161 | 0.9561 | `[0.3875, 0.43, 0.0]` | `[[711, 60, 4], [55, 710, 10], [7, 27, 741]]` |

Best calibrated result: `e311f_lite_e310_morph_r2_r1_cls_only_snr005_ls003` = `0.9376`.

Interpretation: if calibrated/ensemble results exceed the raw single model, the remaining error has a threshold/variance component; if they remain below `0.94`, the visual E3.11f data version is likely capped by class-boundary ambiguity.
