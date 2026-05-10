# E6 Local Counterfactual Benchmark

Date: 2026-05-10

## Purpose

E6 tests whether the raw transformer can use local noise placement while Lead-I SQI summary models fail. The dataset keeps the same clean ECG, same noise kind, and same global SNR inside each counterfactual group, then changes the placement:

- QRS-overlap
- T/ST-overlap
- non-critical region
- uniform whole-segment

The label is no longer a direct global-SNR bin. It combines global SNR, critical-region noise energy, and contaminated beat fraction.

## Dataset

Artifact root:

`outputs/transformer_e6_local_counterfactual`

Rows:

- train: 16,000
- val: 3,200
- test: 3,200
- total: 22,400

Counterfactual groups:

- 5,600 groups
- 5,590 groups contain mixed labels at the same SNR

Sanity checks:

- measured-SNR oracle accuracy: 0.3818
- this passes the key requirement that global SNR no longer determines the label

Test label counts:

- good: 525
- medium: 1,206
- bad: 1,469

## SQI-ML Baseline

Artifact root:

`outputs/transformer_e6_local_counterfactual_sqi_ml`

Feature set:

- Lead I only
- 7 SQI features: iSQI, bSQI, pSQI, sSQI, kSQI, fSQI, basSQI

Test results:

| model | acc | macro F1 | good recall | medium recall | bad recall |
|---|---:|---:|---:|---:|---:|
| SVM-RBF | 0.5663 | 0.3986 | 0.0000 | 0.4768 | 0.8421 |
| MLP | 0.5513 | 0.3927 | 0.0019 | 0.5066 | 0.7842 |

The SQI models mostly collapse good into medium/bad because the label depends on where noise falls, not only global summary statistics.

## Transformer E6

Artifact root:

`outputs/transformer_e6_local_counterfactual/models/e6_local_e3_heads`

Initialization:

- from `outputs/transformer_e3_triplet_k1/models/e3_triplet_tune09/ckpt_best_val.pt`
- added local noise mask head
- added noise type head

Training:

- best val acc: 0.8856 at epoch 11
- test acc: 0.8819

Test confusion matrix, rows=true and cols=pred:

```text
[[ 469,  45,   11],
 [  65, 1031, 110],
 [  12, 135, 1322]]
```

Test recall:

| class | recall |
|---|---:|
| good | 0.8933 |
| medium | 0.8549 |
| bad | 0.8999 |

This passes the main E6 criterion: transformer medium recall is far above Lead-I SQI SVM on the local counterfactual benchmark.

## E9 SQI Teacher Distillation

Artifact root:

`outputs/transformer_e6_local_counterfactual/models/e9_sqi_teacher_e3_heads`

Teacher:

- Lead-I 7-SQI MLP trained on the same E6 dataset
- exported probabilities and normalized SQI targets from `outputs/transformer_e6_local_counterfactual_sqi_ml/teacher_targets/mlp_teacher_targets.csv`

Student changes:

- same E6 transformer setup as above
- added SQI regression head
- added KL distillation loss from teacher probabilities

Training:

- best val acc: 0.8878 at epoch 25
- test acc: 0.8816

Test confusion matrix, rows=true and cols=pred:

```text
[[ 461,  51,   13],
 [  48, 1043, 115],
 [   7, 145, 1317]]
```

Test recall:

| class | recall |
|---|---:|
| good | 0.8781 |
| medium | 0.8648 |
| bad | 0.8965 |

E9 slightly improves best validation accuracy and medium recall, but does not improve overall test accuracy. The teacher is weak on E6, so distillation is not yet a clear win.

## E7 Masked Denoising Pretraining

Pretrain artifact root:

`outputs/transformer_e6_local_counterfactual/pretrain/e7_masked_denoise_from_e3`

Fine-tune artifact root:

`outputs/transformer_e6_local_counterfactual/models/e7_masked_pretrain_e6_ft`

Pretraining:

- initialized from E3 triplet checkpoint
- masked patch ratio: 0.35
- objective: masked noisy-to-clean reconstruction plus full denoise MSE
- best pretrain val loss: 0.0281 at epoch 5

Fine-tuning:

- same E6 fine-tune setup as the E6 baseline
- best val acc: 0.8822 at epoch 19
- test acc: 0.8891

Test confusion matrix, rows=true and cols=pred:

```text
[[ 458,  50,   17],
 [  44, 1013, 149],
 [   4,  91, 1374]]
```

Test recall:

| class | recall |
|---|---:|
| good | 0.8724 |
| medium | 0.8400 |
| bad | 0.9353 |

E7 gives the best overall test accuracy so far. It mainly improves bad recall, while medium recall drops relative to E6/E8/E9.

## E8 Contrastive Severity Pretraining

Pretrain artifact root:

`outputs/transformer_e3_triplet_k1/pretrain/e8_contrastive_from_e3`

Fine-tune artifact root:

`outputs/transformer_e6_local_counterfactual/models/e8_contrastive_pretrain_e6_ft`

Pretraining:

- initialized from E3 triplet checkpoint
- uses good/medium/bad triplets from the E3 triplet dataset
- losses: CE, supervised contrastive severity, ordinal ranking, morphology alignment
- train-holdout validation because the E3 artifact has triplets in train only
- best pretrain val loss: 1.7465 at epoch 4
- last pretrain validation accuracy: 0.9937

Fine-tuning:

- same E6 fine-tune setup as the E6 baseline
- best val acc: 0.8828 at epoch 19
- test acc: 0.8803

Test confusion matrix, rows=true and cols=pred:

```text
[[ 445,  65,   15],
 [  45, 1061, 100],
 [   6, 152, 1311]]
```

Test recall:

| class | recall |
|---|---:|
| good | 0.8476 |
| medium | 0.8798 |
| bad | 0.8924 |

E8 gives the best medium recall so far, but it lowers good recall enough that total test accuracy does not improve.

## Experiment Comparison

| run | test acc | best val acc | good recall | medium recall | bad recall | main effect |
|---|---:|---:|---:|---:|---:|---|
| SQI SVM-RBF | 0.5663 | - | 0.0000 | 0.4768 | 0.8421 | summary SQI fails on local placement |
| SQI MLP | 0.5513 | - | 0.0019 | 0.5066 | 0.7842 | summary SQI fails on local placement |
| E6 local heads | 0.8819 | 0.8856 | 0.8933 | 0.8549 | 0.8999 | strong local transformer baseline |
| E7 masked denoise pretrain | 0.8891 | 0.8822 | 0.8724 | 0.8400 | 0.9353 | best total accuracy |
| E8 contrastive severity pretrain | 0.8803 | 0.8828 | 0.8476 | 0.8798 | 0.8924 | best medium recall |
| E9 SQI teacher distillation | 0.8816 | 0.8878 | 0.8781 | 0.8648 | 0.8965 | no total gain from weak teacher |

## Interpretation

E6 is the strongest evidence so far that the transformer has a real advantage over SQI summary models. On the original global-SNR dataset, SQI and transformer were not comparing the same inductive bias cleanly. On E6, the task requires local temporal information, and the Lead-I SQI SVM/MLP collapse while the transformer remains usable.

The current transformer still overfits: train acc reaches 0.9937 while best val is 0.8856. The best checkpoint is early, epoch 11, before later auxiliary phases help. This suggests the next improvement should focus on representation and regularization, not just more epochs.

E9 confirms that SQI teacher distillation is not the first lever to pull. The SQI teacher transfers some boundary information for medium samples, but it also inherits the teacher's blind spot: local placement is compressed away by summary SQI features.

E7 and E8 split the tradeoff clearly. Masked denoising pretraining is better for overall accuracy and bad/noisy cases. Contrastive severity pretraining is better for medium recall, which means it is shaping the decision boundary in the intended direction, but it currently hurts good recall.

## Recommendation

Most recommended route:

1. Keep E6 as the main benchmark for the transformer-vs-SQI claim.
2. Use E7 masked denoising pretraining as the current best route when optimizing total test accuracy.
3. Use E8 contrastive severity as the next route if the target is medium recall; it needs rebalanced fine-tuning so good recall does not collapse.
4. Treat E9 SQI teacher distillation as lower priority on E6, because the teacher itself is weak on this benchmark.
5. Next experiment should combine E7 pretraining with a lighter E8-style medium-boundary loss during fine-tuning, not as a separate heavy pretrain.

Current best E6 result:

`outputs/transformer_e6_local_counterfactual/models/e7_masked_pretrain_e6_ft`

Current E9 result:

`outputs/transformer_e6_local_counterfactual/models/e9_sqi_teacher_e3_heads`

Current E8 result:

`outputs/transformer_e6_local_counterfactual/models/e8_contrastive_pretrain_e6_ft`
