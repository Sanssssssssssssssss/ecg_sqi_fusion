# E3.9 Small Validations

This report follows the E3.9 result instead of tuning the model. The goal was
to validate whether D1 is stable and whether two tiny data-only patches improve
the good/medium boundary.

Current reference:

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall |
| --- | ---: | ---: | ---: | ---: |
| D1 seed 0 | 0.9465 | 0.9153 | 0.9465 | 0.9777 |

All transformer runs kept the same simple setup:

- `cls_pool=cls`
- `input_mode=raw`
- no rank loss
- no local mask head
- no SQI teacher
- no noise-type head
- no score-aware soft CE

## Experiment A: D1 Seed Robustness

Artifact:

```text
outputs/transformer_e39a_smooth_morph_triplet
```

| Seed | Test Acc | Balanced Acc | Macro F1 | Recall good / medium / bad |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.9465 | 0.9465 | 0.9466 | 0.9153 / 0.9465 / 0.9777 |
| 1 | 0.9505 | 0.9505 | 0.9504 | 0.9287 / 0.9346 / 0.9881 |
| 2 | 0.9416 | 0.9416 | 0.9417 | 0.9227 / 0.9391 / 0.9629 |
| 3 | 0.9485 | 0.9485 | 0.9486 | 0.9376 / 0.9376 / 0.9703 |

Seed robustness:

| Seeds | Mean Test Acc | Population Std | Sample Std |
| --- | ---: | ---: | ---: |
| 1-3 | 0.9468 | 0.0038 | 0.0047 |
| 0-3 | 0.9468 | 0.0033 | 0.0038 |

Decision: D1 passes the robustness check. The proposed success bar was mean
`>= 0.942` and std `<= 0.006`; seeds 1-3 meet both.

## Experiment B: D1 + T-ST Good Guard

New label version:

```text
e39a_smooth_morph_tst_guard
```

Only the good rule changed:

```text
good:
  smooth_morph_score <= 0.10
  qrs_nprd <= 0.10
  tst_nprd <= 0.18
  beat_corr >= 0.95
```

Medium and bad rules were unchanged from E3.9a.

Artifact:

```text
outputs/transformer_e39a_smooth_morph_tst_guard_triplet
```

### Data Audit

| Check | Result | Pass |
| --- | ---: | --- |
| train triplets | 3328 | yes |
| val triplets | 645 | yes |
| test triplets | 649 | yes |
| measured-SNR oracle | 0.3334 | yes |
| max measured-SNR mean gap | 0.0574 dB | yes |
| max proxy pSQI gap | 0.0026 | yes |
| max proxy basSQI gap | 0.0048 | yes |
| observable margin p10 | 0.3477 | yes |
| observable margin p50 | 0.4283 | yes |

SQI baseline:

| Model | Test Acc | Balanced Acc | Macro F1 | Medium Recall |
| --- | ---: | ---: | ---: | ---: |
| SVM-RBF | 0.5419 | 0.5419 | 0.5403 | 0.6102 |
| MLP | 0.5049 | 0.5049 | 0.4994 | 0.5732 |

The data audit passed, so B1 was trained.

### Transformer Result

| Run | Best Val Epoch | Best Val Acc | Test Acc | Balanced Acc | Macro F1 | Recall good / medium / bad |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B1 | 23 | 0.9494 | 0.9389 | 0.9389 | 0.9391 | 0.9199 / 0.9476 / 0.9492 |

Confusion matrix, row order `good / medium / bad`:

```text
[[597, 42, 10],
 [24, 615, 10],
 [3, 30, 616]]
```

Readout: the T-ST guard slightly improved good recall versus D1 seed 0
(`0.9199` vs `0.9153`) and reduced good-to-medium errors (`42` vs `53`), but
it also reduced bad recall sharply (`0.9492` vs `0.9777`). Overall test accuracy
fell from `0.9465` to `0.9389`.

Decision: do not keep B1 as the mainline and do not combine it with other
patches.

## Experiment C: D1 + Score-First Triplet Selection

New label version:

```text
e39a_smooth_morph_scorefirst
```

The label rule is the same as E3.9a. The only change is the ordering inside the
preferred `0.50 dB` SNR pool:

```text
key = (
  abs(smooth_morph_score - target_damage),
  abs(measured_snr_db - matched_snr_db),
)
```

Artifact:

```text
outputs/transformer_e39a_smooth_morph_scorefirst_triplet
```

### Data Audit

| Check | Result | Pass |
| --- | ---: | --- |
| train triplets | 3340 | yes |
| val triplets | 656 | yes |
| test triplets | 673 | yes |
| measured-SNR oracle | 0.3342 | yes |
| max measured-SNR mean gap | 0.2623 dB | no |
| max proxy pSQI gap | 0.0027 | yes |
| max proxy basSQI gap | 0.0040 | yes |
| observable margin p10 | 0.3465 | yes |
| observable margin p50 | 0.4243 | yes |

Measured-SNR class means:

| Class | Mean measured SNR |
| --- | ---: |
| good | 7.5290 |
| medium | 7.5206 |
| bad | 7.2667 |

Readout: score-first selection created a bad-class SNR mean shift even though
the oracle accuracy stayed near chance. The pre-training audit required max SNR
mean gap `<= 0.08 dB`; C1 reached `0.2623 dB`.

Decision: C1 failed audit, so no SQI baseline and no transformer training were
run.

## Overall Decision

Keep E3.9a/D1 as the mainline candidate.

Do not run the combination experiment yet:

- B1 did not improve over D1.
- C1 failed the SNR audit before training.

The useful outcome is that D1 is not a lucky seed. It is stable over seeds
1-3, and the most direct good/medium boundary patch did not beat it. The next
experiment should stay data-side and small, but it needs a better target than
T-ST good guard alone.
