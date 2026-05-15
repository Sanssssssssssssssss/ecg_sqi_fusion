# E3.6 Critical-Damage Triplet

Date: 2026-05-15

## Purpose

E3.6 is a small patch on the E3.5 morphology-damage triplet generator.
It keeps the simple triplet design and matched global SNR, but changes the label
axis from overall morphology damage to critical morphology damage first.

This is an experiment readout, not a new main benchmark.

## Dataset Rule

Artifact:

`outputs/transformer_e36_critical_damage_triplet`

Generator:

```bash
.venv/bin/python -m src.transformer_pipeline.noise.synthesize_morph_damage_triplet \
  --artifact_dir outputs/transformer_e36_critical_damage_triplet \
  --source_artifact_dir outputs/transformer \
  --label_version e36_critical_damage \
  --group_retries 12 \
  --force \
  --verbose
```

RR-level labels:

```bash
.venv/bin/python -m src.transformer_pipeline.noise.make_rr_noise_level \
  --artifact_dir outputs/transformer_e36_critical_damage_triplet \
  --force \
  --verbose
```

Accepted data:

| Split | Triplets | Rows | Good | Medium | Bad |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 3,749 | 11,247 | 3,749 | 3,749 | 3,749 |
| val | 737 | 2,211 | 737 | 737 | 737 |
| test | 758 | 2,274 | 758 | 758 | 758 |

Audit:

- accepted triplet groups: `5,244`
- skipped groups: `356`
- gray-zone candidate rows saved out of benchmark: `46,735`
- measured-SNR oracle accuracy: `0.3343`
- max class mean measured-SNR gap: `0.0262 dB`

## Label Rule

E3.6 stores both:

```text
critical_damage_score =
    0.45 * qrs_nprd
  + 0.35 * tst_nprd
  + 0.15 * (1 - beat_corr)
  + 0.05 * max_beat_nprd
```

and:

```text
global_noise_score = global_nprd
```

The main label is driven by critical damage, not global noise.

Main margins:

- `good`: `critical_damage_score <= 0.12`, `qrs_nprd <= 0.10`, `tst_nprd <= 0.12`, `beat_corr >= 0.94`
- `medium`: bounded QRS and beat correlation, with moderate critical or ST/T damage
- `bad`: `qrs_nprd >= 0.35`, or `beat_corr <= 0.70`, or `critical_damage_score >= 0.55`

Gray zones are excluded from the main three-class benchmark.

Label subtype coverage:

| Class | Subtype | Total |
| --- | --- | ---: |
| good | `good_critical_clean` | 2,220 |
| good | `good_noncritical_hard` | 3,024 |
| medium | `medium_qrs_mild` | 819 |
| medium | `medium_tst_damage` | 2,293 |
| medium | `medium_uniform_moderate` | 2,131 |
| bad | `bad_qrs_damage` | 4,692 |
| bad | `bad_critical_severe` | 551 |

There is also one accepted `bad_low_beat_corr` and one accepted
`medium_critical_damage`; they are negligible edge cases.

## Shortcut Audit

Mean global noise is almost identical across classes:

| Class | Critical Damage Mean | Global Noise Mean |
| --- | ---: | ---: |
| good | 0.0725 | 0.3697 |
| medium | 0.3145 | 0.3698 |
| bad | 0.2918 | 0.3706 |

This is the intended E3.6 behavior: global noise no longer defines the class.
Bad has lower mean critical score than medium because many bad samples are
QRS-margin failures, not high total critical-score failures.

## SQI Baseline

Artifact:

`outputs/transformer_e36_critical_damage_triplet_sqi_ml`

| Model | Test Acc | Balanced Acc | Macro F1 | Medium Recall |
| --- | ---: | ---: | ---: | ---: |
| SQI-SVM | 0.4635 | 0.4635 | 0.4609 | 0.5594 |
| SQI-MLP | 0.4494 | 0.4494 | 0.4446 | 0.5699 |

Readout: E3.6 suppresses the SQI shortcut more strongly than E3.5. This is
good for benchmark cleanliness, but it makes the raw waveform task harder.

## Transformer Runs

R0 references:

| Run | Data | Model | Test Acc |
| --- | --- | --- | ---: |
| R0a | E3.5 | decoder mean, simple E3 recipe | 0.9255 |
| R0b | E3.5 | CLS only, same simple heads | 0.9387 |

E3.6 runs:

| Run | Data | Model / Training | Best Val | Test Acc | Balanced Acc | Macro F1 | Recall good / med / bad |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| R1 | E3.6 | CLS only, no new heads | 0.9218 | 0.8989 | 0.8989 | 0.8992 | 0.8945 / 0.8707 / 0.9314 |
| R2 | E3.6 | CLS only + triplet rank loss `0.05`, margin `0.15` | 0.9132 | 0.9011 | 0.9011 | 0.9010 | 0.9116 / 0.8602 / 0.9314 |

R1 test confusion matrix:

```text
[[678,  75,   5],
 [ 68, 660,  30],
 [  8,  44, 706]]
```

R2 test confusion matrix:

```text
[[691,  54,  13],
 [ 75, 652,  31],
 [ 11,  41, 706]]
```

## Margin-Tuned Follow-Up

The first hard-case audit showed that a small part of the implementation was
not matching the written E3.6 rule: samples in the intended gray zones
`0.12 < critical_damage_score < 0.22` and
`0.42 < critical_damage_score < 0.55` could still enter the `medium` class via
the T-ST condition. Most of these were low-critical T-ST/noncritical cases that
the model reasonably predicted as good.

The generator was patched so gray zones are excluded before medium assignment,
and T-ST medium assignment no longer fires for `noncritical` placement unless
the critical-damage margin itself is satisfied.

Tuned artifact:

`outputs/transformer_e36_critical_damage_triplet_margin_tuned`

Accepted tuned data:

| Split | Triplets | Rows | Good | Medium | Bad |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 3,650 | 10,950 | 3,650 | 3,650 | 3,650 |
| val | 716 | 2,148 | 716 | 716 | 716 |
| test | 738 | 2,214 | 738 | 738 | 738 |

Tuned audit:

- accepted triplet groups: `5,104`
- skipped groups: `496`
- gray-zone candidate rows saved out of benchmark: `76,990`
- measured-SNR oracle accuracy: `0.3352`

Tuned subtype coverage:

| Class | Subtype | Total |
| --- | --- | ---: |
| good | `good_critical_clean` | 2,186 |
| good | `good_noncritical_hard` | 2,918 |
| medium | `medium_qrs_mild` | 873 |
| medium | `medium_tst_damage` | 2,081 |
| medium | `medium_uniform_moderate` | 2,148 |
| bad | `bad_qrs_damage` | 4,663 |
| bad | `bad_critical_severe` | 441 |

Tuned SQI baseline:

| Model | Test Acc | Balanced Acc | Macro F1 | Medium Recall |
| --- | ---: | ---: | ---: | ---: |
| SQI-SVM | 0.5063 | 0.5063 | 0.5024 | 0.6287 |
| SQI-MLP | 0.4512 | 0.4512 | 0.4372 | 0.6450 |

The SVM baseline increases slightly, but remains far below the transformer.
The tuned dataset therefore still suppresses the global SQI shortcut enough for
this experiment.

Tuned transformer runs:

| Run | Data | Model / Training | Best Val | Test Acc | Balanced Acc | Macro F1 | Recall good / med / bad |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| R3 | E3.6 margin-tuned | CLS only, no new heads | 0.9334 | 0.9241 | 0.9241 | 0.9242 | 0.9201 / 0.9011 / 0.9512 |
| R4 | E3.6 margin-tuned | CLS only + triplet rank loss `0.05`, margin `0.15` | 0.9423 | 0.9255 | 0.9255 | 0.9256 | 0.9255 / 0.9119 / 0.9390 |

R3 test confusion matrix:

```text
[[679,  50,   9],
 [ 46, 665,  27],
 [  6,  30, 702]]
```

R4 test confusion matrix:

```text
[[683,  42,  13],
 [ 41, 673,  24],
 [  6,  39, 693]]
```

## Interpretation

E3.6 succeeds as a data-cleanliness probe:

- global SNR and global noise are tightly matched across classes
- SQI baselines drop to roughly 0.45-0.46 test accuracy
- `good_noncritical_hard` is present and balanced enough to test the rule that noise outside critical morphology can still be good
- medium is no longer one unlabeled bucket; it has T-ST, uniform, and mild-QRS subtypes

The first E3.6 pass did not improve the current transformer line:

- R1 is far below the E3.5 CLS result
- R2 gives only +0.0022 test accuracy over R1 and lower validation accuracy
- ranking loss shifts the good/medium boundary a little, but does not solve the generalization gap

The margin-tuned follow-up is more useful:

- enforcing the gray-zone rule gives the main gain: R1 `0.8989` -> R3 `0.9241`
- rank loss helps only after the label boundary is cleaner: R3 `0.9241` -> R4 `0.9255`
- R4 matches the old E3.5 decoder-mean baseline, but still trails the E3.5 CLS result

Conclusion: the best simple path is data semantics first, then a small optional
triplet ranking regularizer. Do not add local heads or larger pooling for this
line yet. The next bottleneck is still label/data definition, not architecture.
