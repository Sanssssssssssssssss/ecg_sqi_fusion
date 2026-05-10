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

## Interpretation

E6 is the strongest evidence so far that the transformer has a real advantage over SQI summary models. On the original global-SNR dataset, SQI and transformer were not comparing the same inductive bias cleanly. On E6, the task requires local temporal information, and the Lead-I SQI SVM/MLP collapse while the transformer remains usable.

The current transformer still overfits: train acc reaches 0.9937 while best val is 0.8856. The best checkpoint is early, epoch 11, before later auxiliary phases help. This suggests the next improvement should focus on representation and regularization, not just more epochs.

## Recommendation

Most recommended route:

1. Keep E6 as the main benchmark for the transformer-vs-SQI claim.
2. Tune E6 class balance and class weights so good is not underrepresented.
3. Try E7 masked/denoising pretraining next, because E6 shows the model benefits from local morphology/noise placement.
4. Treat E9 SQI teacher distillation as lower priority on E6, because the teacher itself is weak on this benchmark.
5. Use E8 contrastive severity only after E7, because it is more invasive and needs careful embedding design.

Current best E6 result:

`outputs/transformer_e6_local_counterfactual/models/e6_local_e3_heads`
