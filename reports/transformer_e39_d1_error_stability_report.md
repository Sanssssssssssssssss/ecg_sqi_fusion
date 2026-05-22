# E3.9 D1 Error Stability

This report freezes E3.9a/D1 as the mainline result and runs diagnostics only.
No data generation, model architecture, ranking loss, local head, teacher, SQI
head, raw-critical input, or score-aware soft CE was changed.

Main result to report:

| Result | Test Acc | Notes |
| --- | ---: | --- |
| D1 seed 0 | 0.9465 | single-model reference |
| D1 seeds 0-3 | 0.9468 +/- 0.0038 | mean +/- sample std |
| SQI-SVM | 0.5364 | global SQI baseline |
| SQI-MLP | 0.5052 | global SQI baseline |

Artifacts:

```text
outputs/transformer_e39a_smooth_morph_triplet
outputs/transformer_e39a_smooth_morph_triplet/diagnostics/d1_error_stability.json
```

## Seed-Level Reference

All four models used the same simple setup:

- `cls_pool=cls`
- `input_mode=raw`
- no rank loss
- no local mask head
- no SQI teacher
- no noise-type head

| Seed | Test Acc | Balanced Acc | Macro F1 | Recall good / medium / bad |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.9465 | 0.9465 | 0.9466 | 0.9153 / 0.9465 / 0.9777 |
| 1 | 0.9505 | 0.9505 | 0.9504 | 0.9287 / 0.9346 / 0.9881 |
| 2 | 0.9416 | 0.9416 | 0.9417 | 0.9227 / 0.9391 / 0.9629 |
| 3 | 0.9485 | 0.9485 | 0.9486 | 0.9376 / 0.9376 / 0.9703 |

Confusion matrices, row order `good / medium / bad`:

```text
seed 0
[[616, 53, 4],
 [22, 637, 14],
 [2, 13, 658]]

seed 1
[[625, 41, 7],
 [32, 629, 12],
 [2, 6, 665]]

seed 2
[[621, 47, 5],
 [28, 632, 13],
 [6, 19, 648]]

seed 3
[[631, 40, 2],
 [28, 631, 14],
 [7, 13, 653]]
```

## Error Consistency

Definitions:

- `always_correct`: all four seeds predict the correct class.
- `sometimes_wrong`: at least one seed is wrong, but not all four.
- `always_wrong`: all four seeds are wrong.

| Group | Count | Rate |
| --- | ---: | ---: |
| always_correct | 1785 | 0.8841 |
| sometimes_wrong | 210 | 0.1040 |
| always_wrong | 24 | 0.0119 |

Readout: only `24 / 2019` test samples are wrong for all four seeds. Most
remaining errors are unstable across seeds, which points to model variance more
than a large block of deterministic label failures.

### By True Class

| True Class | N | Wrong Any | Wrong Any Rate | Always Correct | Sometimes Wrong | Always Wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| good | 673 | 100 | 0.1486 | 573 | 87 | 13 |
| medium | 673 | 90 | 0.1337 | 583 | 81 | 9 |
| bad | 673 | 44 | 0.0654 | 629 | 42 | 2 |

Good and medium are the main uncertainty band. Bad is the most stable class.

### By Majority Prediction

| Majority Pred | N | Wrong Any | Wrong Any Rate | Always Correct | Sometimes Wrong | Always Wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| medium | 696 | 113 | 0.1624 | 583 | 98 | 15 |
| good | 651 | 78 | 0.1198 | 573 | 70 | 8 |
| bad | 672 | 43 | 0.0640 | 629 | 42 | 1 |

The unstable region is mostly the good/medium boundary, not the bad decision.

### True To Majority Prediction

| True Class | Majority Pred | N | Wrong Any | Always Wrong |
| --- | --- | ---: | ---: | ---: |
| good | good | 627 | 54 | 0 |
| good | medium | 44 | 44 | 13 |
| good | bad | 2 | 2 | 0 |
| medium | good | 22 | 22 | 8 |
| medium | medium | 644 | 61 | 0 |
| medium | bad | 7 | 7 | 1 |
| bad | good | 2 | 2 | 0 |
| bad | medium | 8 | 8 | 2 |
| bad | bad | 663 | 34 | 0 |

The persistent error set is small: `13` good-to-medium, `8` medium-to-good, `2`
bad-to-medium, and `1` medium-to-bad.

### By Placement

| Placement | N | Wrong Any | Wrong Any Rate | Always Correct | Sometimes Wrong | Always Wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| noncritical | 676 | 103 | 0.1524 | 573 | 89 | 14 |
| uniform | 438 | 57 | 0.1301 | 381 | 51 | 6 |
| tst_overlap | 271 | 35 | 0.1292 | 236 | 33 | 2 |
| qrs_overlap | 634 | 39 | 0.0615 | 595 | 37 | 2 |

Noncritical good/medium cases remain the hardest area. QRS-overlap bad cases
are comparatively stable.

### By Noise Kind

| Noise Kind | N | Wrong Any | Wrong Any Rate | Always Correct | Sometimes Wrong | Always Wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mix | 516 | 75 | 0.1453 | 441 | 69 | 6 |
| ma | 483 | 59 | 0.1222 | 424 | 56 | 3 |
| bw | 504 | 55 | 0.1091 | 449 | 46 | 9 |
| em | 516 | 45 | 0.0872 | 471 | 39 | 6 |

No single noise kind dominates the total error pool. `mix` has the highest
wrong-any rate, while persistent errors are still spread across noise kinds.

## Metric Bins

### Smooth Morph Score

| Bin | N | Wrong Any | Wrong Any Rate | Always Correct | Sometimes Wrong | Always Wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <0.10 | 673 | 100 | 0.1486 | 573 | 87 | 13 |
| >=0.58 | 40 | 5 | 0.1250 | 35 | 5 | 0 |
| 0.40-0.58 | 65 | 8 | 0.1231 | 57 | 7 | 1 |
| 0.30-0.40 | 825 | 85 | 0.1030 | 740 | 76 | 9 |
| 0.20-0.30 | 416 | 36 | 0.0865 | 380 | 35 | 1 |

The low-score good band has the largest persistent error count, mostly
noncritical samples that still look medium to every seed.

### QRS NPRD

| Bin | N | Wrong Any | Wrong Any Rate | Always Correct | Sometimes Wrong | Always Wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20-0.35 | 252 | 37 | 0.1468 | 215 | 34 | 3 |
| <0.10 | 940 | 133 | 0.1415 | 807 | 118 | 15 |
| 0.10-0.20 | 193 | 25 | 0.1295 | 168 | 21 | 4 |
| >=0.45 | 634 | 39 | 0.0615 | 595 | 37 | 2 |

Severe QRS damage is easy for the transformer. The uncertainty is not driven by
the explicit QRS-bad trigger.

### T-ST NPRD

| Bin | N | Wrong Any | Wrong Any Rate | Always Correct | Sometimes Wrong | Always Wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.18-0.25 | 15 | 3 | 0.2000 | 12 | 3 | 0 |
| 0.25-0.45 | 92 | 17 | 0.1848 | 75 | 15 | 2 |
| >=0.45 | 629 | 80 | 0.1272 | 549 | 73 | 7 |
| <0.12 | 1204 | 127 | 0.1055 | 1077 | 112 | 15 |
| 0.12-0.18 | 79 | 7 | 0.0886 | 72 | 7 | 0 |

The highest T-ST error rates are in small moderate bands, but persistent errors
also appear in low-TST noncritical good samples.

### Beat Correlation

| Bin | N | Wrong Any | Wrong Any Rate | Always Correct | Sometimes Wrong | Always Wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80-0.90 | 3 | 1 | 0.3333 | 2 | 1 | 0 |
| 0.90-0.95 | 15 | 4 | 0.2667 | 11 | 4 | 0 |
| >=0.95 | 2001 | 229 | 0.1144 | 1772 | 205 | 24 |

Almost all samples are in the high-correlation regime, so beat-correlation bins
are not the main explanatory axis for the residual errors.

### Observable Margin

| Bin | N | Wrong Any | Wrong Any Rate | Always Correct | Sometimes Wrong | Always Wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <0.34 | 186 | 40 | 0.2151 | 146 | 35 | 5 |
| 0.48-0.56 | 423 | 49 | 0.1158 | 374 | 44 | 5 |
| >=0.56 | 237 | 27 | 0.1139 | 210 | 24 | 3 |
| 0.34-0.40 | 498 | 52 | 0.1044 | 446 | 47 | 5 |
| 0.40-0.48 | 675 | 66 | 0.0978 | 609 | 60 | 6 |

Low observable margin is clearly harder, but it does not explain all errors.
The best interpretation is mixed: part of the error is true visual ambiguity,
and part is seed variance.

## Validation Logit-Offset Calibration

Calibration searches class offsets on validation:

```text
logits_calibrated = logits + [b_good, b_medium, b_bad]
```

The search fixes `b_bad=0`, because adding a constant to all logits does not
change predictions.

| Seed | Raw Test Acc | Objective | Offsets good / medium / bad | Val Acc | Test Acc | Test Recall good / medium / bad |
| ---: | ---: | --- | --- | ---: | ---: | --- |
| 0 | 0.9465 | acc/bal_acc | 0.92 / -1.06 / 0.00 | 0.9548 | 0.9416 | 0.9302 / 0.9168 / 0.9777 |
| 1 | 0.9505 | acc/bal_acc | 0.25 / -1.00 / 0.00 | 0.9548 | 0.9510 | 0.9376 / 0.9242 / 0.9911 |
| 2 | 0.9416 | accuracy | -0.70 / -0.65 / 0.00 | 0.9482 | 0.9421 | 0.9212 / 0.9361 / 0.9688 |
| 2 | 0.9416 | balanced_accuracy | -0.60 / -0.53 / 0.00 | 0.9487 | 0.9425 | 0.9212 / 0.9376 / 0.9688 |
| 3 | 0.9485 | acc/bal_acc | -0.20 / 0.25 / 0.00 | 0.9553 | 0.9480 | 0.9346 / 0.9391 / 0.9703 |

Readout: validation calibration does not solve D1. It hurts seed 0, barely
helps seeds 1-2, and slightly hurts seed 3. Do not use calibrated offsets as
the main result.

## Seed Ensemble

Four-seed probability averaging:

```text
p_ensemble = mean(softmax(logits_seed0..3))
```

| Split | Acc | Balanced Acc | Macro F1 | Recall good / medium / bad |
| --- | ---: | ---: | ---: | --- |
| val | 0.9670 | 0.9670 | 0.9670 | 0.9512 / 0.9665 / 0.9832 |
| test | 0.9619 | 0.9619 | 0.9619 | 0.9450 / 0.9525 / 0.9881 |

Test confusion matrix, row order `good / medium / bad`:

```text
[[636, 37, 0],
 [25, 641, 7],
 [1, 7, 665]]
```

Readout: the ensemble crosses `0.96` without changing the data or model class.
This strongly suggests the single-model ceiling is being limited by seed
variance around the good/medium boundary. The ensemble is useful as a diagnostic
upper bound, but the main reported model should remain the single-model D1
multi-seed result.

## Decision

Keep D1 as the fixed mainline:

- single seed 0: `0.9465`
- four-seed mean: `0.9468 +/- 0.0038`
- SQI baselines stay low: SVM `0.5364`, MLP `0.5052`
- ensemble diagnostic upper bound: `0.9619`

Do not continue the rejected data/model branches right now:

- no T-ST guard + score-first combination
- no score-first continuation
- no E3.7/E3.8 max-score line
- no rank loss
- no soft CE
- no local head
- no teacher distillation
- no raw-critical input
- no noise-type head

The next useful step is reporting D1 as the robust main result and using the
ensemble result only to explain that the residual errors are mostly variance at
the good/medium boundary.
