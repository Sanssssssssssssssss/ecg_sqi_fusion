# E3.5 Morphology-Damage Triplet

Date: 2026-05-15

## Purpose

This is a side experiment, not the new main line.

The goal is to keep the simple E3 triplet shape while replacing the label source.
E3 labels were driven by global SNR bins. E3.5 labels are driven by morphology
damage between clean and synthetic noisy ECG.

## Dataset Rule

For each accepted clean segment, generate exactly three variants:

- same clean ECG
- same noise kind
- same noise window
- matched global SNR in the `6-12 dB` range
- one margin-separated `good`, `medium`, and `bad` sample

The full factorial local branch has been removed from the code path. This
experiment is a compact triplet read.

## Label Rule

Damage score:

```text
0.45 * qrs_nprd
+ 0.25 * tst_nprd
+ 0.20 * (1 - beat_corr)
+ 0.10 * max_beat_nprd
```

Margin labels:

- `good`: `damage_score <= 0.12`, `qrs_nprd <= 0.10`, `beat_corr >= 0.95`
- `medium`: `0.25 <= damage_score <= 0.40`, `qrs_nprd < 0.35`, `beat_corr >= 0.80`
- `bad`: `damage_score >= 0.55`, or `qrs_nprd >= 0.45`, or `beat_corr <= 0.70`

Gray-zone candidates are excluded from the main benchmark and saved only as a
candidate audit table.

## Implementation

Generator:

```bash
.venv/bin/python -m src.transformer_pipeline.noise.synthesize_morph_damage_triplet \
  --artifact_dir outputs/transformer_e35_morph_damage_triplet \
  --source_artifact_dir outputs/transformer \
  --force \
  --verbose
```

Then generate RR-level pseudo noise labels:

```bash
.venv/bin/python -m src.transformer_pipeline.noise.make_rr_noise_level \
  --artifact_dir outputs/transformer_e35_morph_damage_triplet \
  --force \
  --verbose
```

Planned first transformer read:

```bash
ARTIFACT_DIR=outputs/transformer_e35_morph_damage_triplet \
EXPERIMENT_NAME=e35_morph_triplet_e3_simple \
EPOCHS=26 \
BATCH_SIZE=32 \
DROPOUT=0.05 \
E_CLS=0 \
E_DENOISE=18 \
E_LEVEL=4 \
E_UNCERT=4 \
BAD_DEN_W_MAX=0.02 \
LAMBDA_CLS=22 \
LAMBDA_DEN=25 \
LABEL_SMOOTHING=0.02 \
CLASS_WEIGHT_MEDIUM=1.15 \
UNCERTAINTY_MODE=fixed \
sbatch slurm/tune_transformer.sh
```

## Audit To Read

The generator writes:

`outputs/transformer_e35_morph_damage_triplet/datasets/morph_damage_triplet_summary.json`

Key checks:

- class counts should be exactly balanced within each accepted split
- measured SNR means should be close across classes
- proxy `pSQI`, `basSQI`, `kSQI`, `sSQI`, and RMS should not expose a trivial class shortcut
- damage metrics should separate classes by construction

## Dataset Result

Artifact:

`outputs/transformer_e35_morph_damage_triplet`

Generation accepted:

| Split | Clean groups | Rows | Good | Medium | Bad |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 2,959 | 8,877 | 2,959 | 2,959 | 2,959 |
| val | 598 | 1,794 | 598 | 598 | 598 |
| test | 604 | 1,812 | 604 | 604 | 604 |

Audit read:

- accepted triplet groups: `4,161`
- skipped groups: `1,439`
- gray-zone candidate rows excluded from benchmark: `129,045`
- measured-SNR oracle accuracy: `0.3334`
- max class mean measured-SNR gap: `0.0418 dB`
- split-specific measured-SNR class mean gaps: train `0.0427 dB`, val `0.0374 dB`, test `0.0416 dB`
- triplet integrity: `4,161/4,161` groups have exactly 3 rows, 3 labels, the same clean segment, the same noise kind, and the same noise window
- within-triplet measured-SNR spread: max `0.5000 dB`, P90 `0.2500 dB`

Mean measured SNR by class:

| Class | Mean dB | P10 | P50 | P90 |
| --- | ---: | ---: | ---: | ---: |
| good | 7.5708 | 6.3777 | 7.4647 | 8.8686 |
| medium | 7.5613 | 6.3703 | 7.4668 | 8.8074 |
| bad | 7.5290 | 6.3524 | 7.4167 | 8.7754 |

This passes the main data-design check: global SNR is matched tightly enough
that a measured-SNR oracle is essentially chance.

Max class-mean gaps for global shortcut audit:

| Metric | Max class mean gap |
| --- | ---: |
| measured SNR | `0.0418 dB` |
| proxy pSQI | `0.0026` |
| proxy basSQI | `0.0062` |
| proxy kSQI | `2.8382` |
| proxy sSQI | `0.2359` |
| proxy RMS | `0.0001` |

Mean morphology metrics:

| Class | Damage | QRS NPRD | ST/T NPRD | Beat Corr | Max Beat NPRD |
| --- | ---: | ---: | ---: | ---: | ---: |
| good | `0.0917` | `0.0410` | `0.0865` | `0.9976` | `0.5115` |
| medium | `0.3220` | `0.1414` | `0.7518` | `0.9952` | `0.6946` |
| bad | `0.3477` | `0.4979` | `0.1477` | `0.9918` | `0.8504` |

This is intentional: medium is mostly ST/T morphology damage, while bad is
mostly QRS/beat-level damage. The bad class is not required to have high total
damage score when the QRS-specific margin is already violated.

## SQI-ML Baseline

Artifact:

`outputs/transformer_e35_morph_damage_triplet_sqi_ml`

| Model | Test Acc | Balanced Acc | Macro F1 | Medium Recall |
| --- | ---: | ---: | ---: | ---: |
| SQI-SVM | 0.5265 | 0.5265 | 0.5264 | 0.5530 |
| SQI-MLP | 0.4912 | 0.4912 | 0.4894 | 0.5447 |

Readout:

- The SQI baselines are far below the desired 0.96-style target.
- They do not collapse to zero medium recall like E6b, but overall accuracy is near 0.5 despite perfectly balanced classes.
- This is acceptable for the side experiment: morphology damage is not recoverable from the 7 global SQI summary features alone.

## Submitted Transformer Read

First read uses the E3 simple `tune09` recipe:

- raw ECG input
- decoder mean pooling
- no ordinal/SNR/local/noise-type/SQI heads
- 26 epochs
- selected by validation accuracy

Job:

| Job | Experiment | Status |
| --- | --- | --- |
| `29380469` | `e35_morph_triplet_e3_simple` | cancelled before start; resubmitted with shorter walltime |
| `29381015` | `e35_morph_triplet_e3_simple` | cancelled after the `intr` job was accepted |
| `29381953` | `e35_morph_triplet_e3_simple` | completed on `gpu-q-22` |

Queue note:

- The original regular-QOS job had a very late estimated start time, so an
  `intr` QoS job was submitted under the same `mphil-dis-sl2-gpu` account.
- `29381953` started immediately on A100 and wrote logs to
  `logs/ptbxl_tune_29381953.out` and `logs/ptbxl_tune_29381953.err`.

## Transformer Result

Artifact:

`outputs/transformer_e35_morph_damage_triplet/models/e35_morph_triplet_e3_simple`

Best checkpoint:

- selection: validation accuracy
- best epoch: `20`
- best validation accuracy: `0.9264`
- test accuracy: `0.9255`
- test balanced accuracy: `0.9255`
- test macro F1: `0.9256`

Test confusion matrix, rows=true and cols=pred:

```text
[[544,  49,  11],
 [ 27, 560,  17],
 [  8,  23, 573]]
```

Test recall:

| Class | Recall | Precision | F1 |
| --- | ---: | ---: | ---: |
| good | 0.9007 | 0.9396 | 0.9197 |
| medium | 0.9272 | 0.8861 | 0.9061 |
| bad | 0.9487 | 0.9534 | 0.9510 |

Comparison on the same E3.5 test set:

| Model | Test Acc | Balanced Acc | Macro F1 | Medium Recall |
| --- | ---: | ---: | ---: | ---: |
| SQI-SVM | 0.5265 | 0.5265 | 0.5264 | 0.5530 |
| SQI-MLP | 0.4912 | 0.4912 | 0.4894 | 0.5447 |
| Transformer | 0.9255 | 0.9255 | 0.9256 | 0.9272 |

Readout:

- The first E3.5 transformer read does not reach the desired `0.96+` target.
- It strongly beats the 7-SQI summary baselines under matched global SNR.
- The remaining error is mostly good/medium boundary confusion: `49` good samples go to medium, and `27` medium samples go to good.
- Bad/QRS-damage recognition is strongest, with bad recall `0.9487`.

## Error Audit

Audit artifact:

`outputs/transformer_e35_morph_damage_triplet/medium_error_audit/e35_morph_triplet_e3_simple`

Main findings:

- measured-SNR oracle remains chance on val/test: `0.3333`.
- medium test errors: `27` medium -> good, `17` medium -> bad.
- placement accuracy on test: noncritical `0.8979`, uniform `0.9239`, T/ST overlap `0.9320`, QRS overlap `0.9537`.
- noise-kind accuracy on test: bw `0.9005`, ma `0.9161`, mix `0.9408`, em `0.9420`.
- validation-only class-bias calibration barely changes test accuracy: `0.9255` -> `0.9260`.

Interpretation:

The residual errors are not fixed by a simple class-threshold shift. The hard
cases are mostly high-SNR local-boundary cases, especially noncritical good
segments predicted as medium and some medium uniform/T-ST cases pushed into a
neighboring class.

## Follow-Up: CLS Aggregation

The next read keeps the E3.5 data and the E3 simple recipe unchanged, and only
changes the classifier aggregation from decoder mean pooling to a learnable CLS
token.

Reason:

- the earlier factorial-local structure ablation found `CLS only` to be the
  best small structural change;
- E3.5 residual errors are mostly good/medium boundary errors, so a cleaner
  aggregation read is lower risk than adding new heads or local masks;
- this keeps the code path simple and does not change the label rule.

Job:

| Job | Experiment | Change | Status |
| --- | --- | --- | --- |
| `29385952` | `e35_morph_triplet_cls_only` | `cls_pool=cls` only | failed immediately because CLI did not yet allow `cls` |
| `29386149` | `e35_morph_triplet_cls_only` | `cls_pool=cls` only | cancelled while pending; replaced by shorter walltime job |
| `29386207` | `e35_morph_triplet_cls_only` | `cls_pool=cls` only | completed in `00:09:46` |

Minimal code change:

- add `cls_pool=cls` as an optional pooling mode;
- prepend one learnable CLS token before the encoder;
- keep denoise/level/local heads on patch tokens only, so reconstruction length remains `1250`;
- keep default `decoder` pooling unchanged.

Result:

| Model | Pooling | Test Acc | Balanced Acc | Macro F1 | Good Recall | Medium Recall | Bad Recall | Best Val Acc | Best Epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E3.5 simple | decoder mean | 0.9255 | 0.9255 | 0.9256 | 0.9007 | 0.9272 | 0.9487 | 0.9264 | 20 |
| E3.5 CLS | CLS | 0.9387 | 0.9387 | 0.9389 | 0.9470 | 0.9255 | 0.9437 | 0.9292 | 22 |

CLS test confusion matrix, rows=true and cols=pred:

```text
[[572,  26,   6],
 [ 34, 559,  11],
 [  8,  26, 570]]
```

Comparison to decoder mean:

- total errors: `135` -> `111`
- good -> medium: `49` -> `26`
- medium -> good: `27` -> `34`
- medium -> bad: `17` -> `11`
- bad -> medium: `23` -> `26`

Placement accuracy on test:

| Placement | Decoder Mean | CLS |
| --- | ---: | ---: |
| noncritical | 0.8979 | 0.9423 |
| qrs_overlap | 0.9537 | 0.9590 |
| tst_overlap | 0.9320 | 0.9000 |
| uniform | 0.9239 | 0.9289 |

Noise-kind accuracy on test:

| Noise Kind | Decoder Mean | CLS |
| --- | ---: | ---: |
| bw | 0.9005 | 0.9236 |
| em | 0.9420 | 0.9358 |
| ma | 0.9161 | 0.9297 |
| mix | 0.9408 | 0.9649 |

Validation-only class-bias calibration:

| Model | Val Acc | Calibrated Val Acc | Test Acc | Calibrated Test Acc |
| --- | ---: | ---: | ---: | ---: |
| decoder mean | 0.9264 | 0.9281 | 0.9255 | 0.9266 |
| CLS | 0.9292 | 0.9309 | 0.9387 | 0.9387 |

Readout:

- CLS is a real improvement for E3.5, but still not a `0.96+` result.
- The gain comes mostly from fewer good samples being pushed into medium and
  better noncritical/noise-kind robustness.
- Medium recall is essentially unchanged, so the remaining issue is still the
  medium boundary, not a simple aggregation failure.
- Simple class-bias calibration does not improve CLS test accuracy.
