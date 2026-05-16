# E3.8 Core-Diagnostic Triplet

E3.8 is a small rollback-style patch after E3.7 failed. It keeps the E3.7
`diagnostic_damage_score` only as an audit field and adds a simpler label axis:

```text
core_diagnostic_score = max(qrs_axis, tst_axis, crit_axis, corr_axis)
```

The important change is that `beat_axis` is no longer allowed to define the
main class. It is still saved for audit, together with `dominant_axis`, but the
main label is based on QRS, T-ST, critical damage, and beat-template correlation.

## Generator Patch

Changed file:

- `src/transformer_pipeline/noise/synthesize_morph_damage_triplet.py`

New label version:

- `e38_core_diagnostic_damage`

Saved label columns:

- `diagnostic_damage_score`: audit only, includes the weighted beat axis
- `core_diagnostic_score`: main E3.8 label axis, excludes beat axis
- `beat_axis`: audit only
- `dominant_axis`: audit only

Label rule:

```text
good:
  core <= 0.32
  qrs_nprd <= 0.10
  tst_nprd <= 0.12
  beat_corr >= 0.94

medium:
  0.45 <= core <= 0.85
  qrs_nprd < 0.35
  beat_corr >= 0.75
  dominant_axis != beat

bad:
  core >= 1.00

gray:
  otherwise
```

The medium range was widened from the first proposed `0.48-0.82` to
`0.45-0.85` to keep enough accepted triplets, while still preventing beat-axis
dominance from becoming a medium shortcut.

Triplet selection for E3.8 uses `core_diagnostic_score` as the class severity
axis and tightens final measured-SNR matching:

- candidate pool: prefer samples within `0.20 dB` of matched SNR, fallback to
  `0.35 dB`
- final triplet measured-SNR gap: `<= 0.25 dB`
- class target severities: good `0.22`, medium `0.68`, bad `1.12`

## Data Audit

Artifact:

```text
outputs/transformer_e38_core_diagnostic_triplet
```

Generation command:

```bash
.venv/bin/python -m src.transformer_pipeline.noise.synthesize_morph_damage_triplet \
  --artifact_dir outputs/transformer_e38_core_diagnostic_triplet \
  --source_artifact_dir outputs/transformer \
  --label_version e38_core_diagnostic_damage \
  --group_retries 24 --force --verbose
```

Accepted triplets:

| Split | Triplets | Rows | Class Counts |
| --- | ---: | ---: | --- |
| train | 3589 | 10767 | 3589 / class |
| val | 713 | 2139 | 713 / class |
| test | 716 | 2148 | 716 / class |
| total | 5018 | 15054 | 5018 / class |

Hard audit checks:

| Check | Result |
| --- | ---: |
| measured-SNR oracle accuracy | 0.3329 |
| max measured-SNR class mean gap | 0.0221 dB |
| max proxy pSQI class mean gap | 0.0008 |
| max proxy basSQI class mean gap | 0.0030 |
| medium + noncritical share of medium | 0.86% |
| medium beat-axis dominant | 0.00% |

Core score separation:

| Class | p10 | p50 | p90 |
| --- | ---: | ---: | ---: |
| good | 0.0863 | 0.1508 | 0.2390 |
| medium | 0.6595 | 0.8009 | 0.8423 |
| bad | 1.0873 | 1.4179 | 2.2114 |

Global noise remained matched:

| Class | measured SNR mean | global_noise_score mean |
| --- | ---: | ---: |
| good | 11.0412 | 0.2826 |
| medium | 11.0589 | 0.2820 |
| bad | 11.0368 | 0.2827 |

Readout: the E3.8 data patch fixed the two E3.7 audit failures. Medium is no
longer dominated by noncritical placement, and beat-axis dominance is removed
from medium.

## SQI Baseline

Command:

```bash
.venv/bin/python -m src.transformer_pipeline.sqi_ml_multiclass \
  --transformer_artifact_dir outputs/transformer_e38_core_diagnostic_triplet \
  --out_dir outputs/transformer_e38_core_diagnostic_triplet_sqi_ml \
  --force --verbose
```

| Model | Features | Test Acc | Balanced Acc | Macro F1 | Medium Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| SVM-RBF | 7 SQI | 0.4334 | 0.4334 | 0.4325 | 0.5000 |
| MLP | 7 SQI | 0.4232 | 0.4232 | 0.4185 | 0.5028 |

Readout: E3.8 still suppresses the global SQI shortcut. The task is not
solvable by the seven SQI summaries alone.

## Transformer Runs

Both runs used the simple CLS/raw setup:

- `cls_pool=cls`
- `input_mode=raw`
- no rank loss
- no local mask head
- no noise type head
- no SQI head
- no teacher distillation

| Run | Data | Method | Best Val | Test Acc | Balanced Acc | Macro F1 | Recall good / medium / bad |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| C1 | E3.8 | CLS raw from scratch | 0.8962 | 0.8794 | 0.8794 | 0.8796 | 0.8911 / 0.8631 / 0.8841 |
| C2 | E3.8 | CLS raw + E3.5 CLS warm-start | 0.9032 | 0.8859 | 0.8859 | 0.8859 | 0.9008 / 0.8617 / 0.8953 |

C1 confusion matrix:

```text
[[638, 42, 36],
 [40, 618, 58],
 [20, 63, 633]]
```

C2 confusion matrix:

```text
[[645, 45, 26],
 [46, 617, 53],
 [23, 52, 641]]
```

## Comparison

| Run | Test Acc | Note |
| --- | ---: | --- |
| E3.5 CLS | 0.9387 | strongest simple CLS result so far |
| E3.6 margin-tuned R3 | 0.9241 | CLS, no rank |
| E3.6 margin-tuned R4 | 0.9255 | CLS + small rank |
| E3.7 B2 | 0.7702 | failed because medium became beat/noncritical dominated |
| E3.8 C1 | 0.8794 | audit fixed, transformer still weaker |
| E3.8 C2 | 0.8859 | warm-start helps slightly, still below E3.6 |

## Conclusion

E3.8 succeeded as a data audit patch but failed as the next benchmark line.
It removed the E3.7 beat-axis failure and kept SQI baselines low, but the raw
transformer only reached `0.8859` test accuracy with warm-start. That is well
below E3.6 margin-tuned R4 (`0.9255`) and far below E3.5 CLS (`0.9387`).

Decision: do not continue tuning E3.8. Use E3.6 margin-tuned as the fallback
critical-damage line, and keep E3.5 CLS as the strongest simple-model reference.
