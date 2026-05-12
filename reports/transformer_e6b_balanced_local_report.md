# E6b Balanced Local Counterfactual Benchmark

Date: 2026-05-11

## Purpose

E6b keeps the E6 local-counterfactual idea but makes the benchmark fairer for final reporting. E6 current remains unchanged and should still be used as the proof-of-concept result. E6b adds:

- more balanced good / medium / bad class counts
- exact split-level stratification for noise kind, placement, and SNR profile
- complete 5 x 4 variants for every selected clean ECG segment
- absolute critical-quality labels instead of the earlier ratio score

## Dataset

Artifact root:

`outputs/transformer_e6b_balanced_local`

Generation:

- clean train segments: 1,000
- clean val segments: 250
- clean test segments: 250
- variants per clean segment: 20
- SNR profiles: -2, 4, 12, 16, 20 dB
- placements: QRS overlap, T/ST overlap, noncritical, uniform
- noise kinds: em, ma, bw, mix

Rows:

| split | rows | good | medium | bad | class balance ratio |
|---|---:|---:|---:|---:|---:|
| train | 20,000 | 7,398 | 5,604 | 6,998 | 1.32 |
| val | 5,000 | 1,821 | 1,431 | 1,748 | 1.27 |
| test | 5,000 | 1,827 | 1,431 | 1,742 | 1.28 |

Stratification checks:

- every clean group has all 20 variants: 1,500 / 1,500 groups complete
- each split has exactly balanced placement counts
- each split has exactly balanced SNR profile counts
- train has exactly balanced noise-kind counts; val/test differ by only 20 samples because 250 clean groups is not divisible by 4
- measured-SNR oracle accuracy: 0.5968
- valid RR rows after noise-level generation: 29,967 / 30,000

## Label Rule

E6b computes:

- `global_snr_db`
- `critical_snr_db`
- `min_beat_critical_snr_db`
- `qrs_snr_db`
- `tst_snr_db`
- `contaminated_beat_fraction`
- `max_consecutive_contaminated_beats`

The label is based on absolute local quality:

- bad if QRS/beat-level critical quality is strongly degraded
- medium if critical quality is borderline or some beats are contaminated
- good only when global and critical quality are both high and beat contamination is low

This keeps high-SNR uniform noise from being over-penalized while still marking local QRS or T/ST contamination as clinically worse.

## SQI-ML Baseline

Artifact root:

`outputs/transformer_e6b_balanced_local_sqi_ml`

Feature set:

- Lead I only
- 7 SQI features: iSQI, bSQI, pSQI, sSQI, kSQI, fSQI, basSQI

Test results:

| model | acc | balanced acc | macro F1 | good recall | medium recall | bad recall |
|---|---:|---:|---:|---:|---:|---:|
| SVM-RBF | 0.5824 | 0.5448 | 0.4580 | 0.7822 | 0.0056 | 0.8467 |
| MLP | 0.5760 | 0.5390 | 0.4539 | 0.7674 | 0.0070 | 0.8427 |

SVM-RBF confusion matrix, rows=true and cols=pred:

```text
[[1429,    4,  394],
 [ 962,    8,  461],
 [ 259,    8, 1475]]
```

MLP confusion matrix, rows=true and cols=pred:

```text
[[1402,   10,  415],
 [ 928,   10,  493],
 [ 259,   15, 1468]]
```

## Transformer Result

Artifact root:

`outputs/transformer_e6b_balanced_local/models/e7_masked_pretrain_e6b_ft`

Setup:

- initialized from E6b masked denoise pretraining
- local mask head enabled
- noise type head enabled
- fixed loss weights
- selected checkpoint by best validation accuracy

Pipeline status:

| step | status |
|---|---|
| forward_check | done |
| train | done |
| evaluate | done |

Validation/test results:

| split | acc | balanced acc | macro F1 | good recall | medium recall | bad recall |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.7988 | 0.7896 | 0.7886 | 0.7886 | 0.6450 | 0.9354 |
| test | 0.7906 | 0.7799 | 0.7785 | 0.7904 | 0.6108 | 0.9386 |

Best validation checkpoint:

- epoch: 7
- phase: B_add_denoise
- validation accuracy: 0.7988

Transformer test confusion matrix, rows=true and cols=pred:

```text
[[1444, 287,  96],
 [ 383, 874, 174],
 [  34,  73, 1635]]
```

Compared with Lead-I SQI baselines on the same E6b test set:

| model | test acc | balanced acc | macro F1 | medium recall |
|---|---:|---:|---:|---:|
| SVM-RBF SQI | 0.5824 | 0.5448 | 0.4580 | 0.0056 |
| MLP SQI | 0.5760 | 0.5390 | 0.4539 | 0.0070 |
| Transformer | 0.7906 | 0.7799 | 0.7785 | 0.6108 |

## Transformer Architecture Layer 2

Artifact root:

`outputs/transformer_e6b_balanced_local/models/e6b_arch2_pos_localpool`

Changes:

- encoder positional embedding: `tok = tok + pos_enc`
- decoder positional embedding: `z = dec_in(h) + pos_dec`
- classification pooling changed from mean pooling to local-aware pooling:
  - mean over tokens
  - max over tokens
  - top-k token pooling, k=8
  - local severity attention from the local mask head
- the local mask head is now structurally coupled to classification, not only an auxiliary loss
- old E6b checkpoint was warm-started with compatible weights; new positional embeddings and widened classifier heads were initialized fresh

Validation/test results:

| model | test acc | balanced acc | macro F1 | good recall | medium recall | bad recall | best val |
|---|---:|---:|---:|---:|---:|---:|---:|
| E6b transformer baseline | 0.7906 | 0.7799 | 0.7785 | 0.7904 | 0.6108 | 0.9386 | 0.7988 |
| E6b arch layer 2 | 0.7960 | 0.7863 | 0.7862 | 0.8062 | 0.6359 | 0.9168 | 0.8010 |

Layer 2 test confusion matrix, rows=true and cols=pred:

```text
[[1473, 266,   88],
 [ 402, 910,  119],
 [  21, 124, 1597]]
```

Pipeline status:

| step | status |
|---|---|
| forward_check | done |
| train | done |
| evaluate | done |

Interpretation:

Layer 2 gives a small but real improvement on E6b: +0.0054 test accuracy, +0.0064 balanced accuracy, +0.0076 macro F1, and +0.0252 medium recall. The tradeoff is lower bad recall, from 0.9386 to 0.9168, because the model shifts some bad samples toward medium.

The learning curve also shows fast overfitting. Best validation accuracy happens at epoch 10, while later train accuracy rises above 0.98 and validation falls. This suggests the local-aware architecture is useful, but the next step should control the decision boundary rather than simply training longer.

## Interpretation

E6b removes the main critique that E6 current may be unfair because good is rare. Even with balanced class counts, balanced placements, balanced SNR profiles, and balanced noise kinds, Lead-I 7-SQI SVM/MLP still almost completely fail to recover the medium class.

The transformer is not yet a final 0.95-style model on E6b, but it is already much better aligned with the local benchmark. The baseline transformer medium recall is 0.6108, and the local-aware layer 2 model improves it to 0.6359, while SQI-SVM/MLP are effectively zero. This supports the central claim of E6/E6b: once label quality depends on local temporal placement rather than only global SNR or SQI summaries, raw sequence models can use information that summary-SQI models lose.

The remaining weakness is still the medium boundary. Most transformer medium errors go to good, not bad, so the next useful experiment is not another SQI baseline. It should improve the transformer objective around borderline local contamination, for example with ordinal/SNR auxiliary supervision or calibrated medium thresholds on the validation set.

## Current Status

Completed:

- E6b dataset generation
- RR noise-level generation
- transformer dry-run forward/training input check
- Lead-I SQI SVM/MLP baseline
- E6b transformer pretraining, fine-tuning, and evaluation
- E6b layer 2 local-aware transformer fine-tuning and evaluation

Next:

- tune the E6b transformer medium boundary with ordinal/SNR supervision
- optionally calibrate validation thresholds for medium vs neighboring classes
