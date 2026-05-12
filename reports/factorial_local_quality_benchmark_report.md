# Factorial Local Quality Benchmark Report

## Purpose

This branch replaces the E6/E6b naming with a clearer benchmark line: **Factorial Local Quality Benchmark**.

The benchmark tests whether a raw ECG transformer can use local contamination context, not just global SNR or global SQI summaries.

## Data Design

- Source: PTB-XL Lead I 10 s clean segments from `outputs/transformer`.
- Split rule: full counterfactual groups stay inside one `train` / `val` / `test` split by clean ECG segment.
- Factors per clean segment:
  - `noise_kind`: `em`, `ma`, `bw`, `mix`
  - `snr_profile`: `-2`, `4`, `8`, `12`, `20` dB
  - `placement`: `qrs_overlap`, `tst_overlap`, `noncritical`, `uniform`
- Variants per clean segment: `4 x 5 x 4 = 80`.
- First trial size: `120/40/40` clean segments for train/val/test, producing `16000` total samples.

## Dataset Sanity

- Complete factorial groups: `200/200`
- Measured-SNR oracle accuracy: `0.5538`
- This confirms the new labels are not recoverable from global measured SNR alone.

Class counts:

| Split | Good | Medium | Bad |
| --- | ---: | ---: | ---: |
| Train | 2529 | 2791 | 4280 |
| Val | 811 | 924 | 1465 |
| Test | 816 | 938 | 1446 |

## SQI-ML Baseline

Three-class Lead-I SQI baselines use the same 7 SQI feature adapter as the existing transformer-dataset comparison.

| Model | Test Acc | Balanced Acc | Macro F1 | Medium Recall |
| --- | ---: | ---: | ---: | ---: |
| SVM-RBF | 0.5528 | 0.4970 | 0.4784 | 0.1780 |
| MLP | 0.5513 | 0.5028 | 0.4672 | 0.1247 |

Interpretation: global SQI summaries collapse on the local placement task, especially on `medium`.

## Transformer Reference

I ran one short in-branch GPU training job and one transfer sanity check.

Short in-branch GPU training:

- Slurm job: `29262317`
- Experiment: `factorial_local_t0_aux_short`
- Epochs: `6`
- Heads: CE + ordinal + SNR + local mask + noise type
- Result: improved over SQI baselines, but underfit compared with the old E6b transfer checkpoint.
- I cancelled the initial 12-epoch job `29261769` to avoid occupying extra GPU quota while the short backfill trial was sufficient for a first read.

Transfer sanity check:

- Model: `outputs/transformer_e6b_balanced_local/models/e7_masked_pretrain_e6b_ft`
- This checkpoint has no local-aware architecture parameters, so it loads on the clean f17ec43 model line.

| Model | Test Acc | Balanced Acc | Macro F1 | Medium Recall | Held-out noise=ma Acc | Held-out SNR=8dB Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SQI-SVM | 0.5528 | 0.4970 | 0.4784 | 0.1780 | 0.5550 | 0.4688 |
| Factorial short train | 0.6525 | 0.6524 | 0.6324 | 0.4200 | 0.6925 | 0.4203 |
| E6b transfer transformer | 0.7863 | 0.7710 | 0.7689 | 0.6450 | 0.8163 | 0.5797 |

Counterfactual/group metrics:

| Metric | SQI-SVM | Factorial short train | E6b transfer |
| --- | ---: | ---: | ---: |
| QRS-overlap worse than noncritical ranking | 0.1088 | 0.8750 | 0.9188 |
| T/ST-overlap worse than noncritical ranking | 0.1188 | 0.8125 | 0.9025 |
| Low-to-high SNR monotonicity | n/a | 0.9773 | 0.9394 |
| Local mask AUPRC | n/a | 0.4503 | 0.6930 |
| Local mask IoU @ 0.5 | n/a | 0.0672 | 0.3529 |

This is the key signal: even a short raw transformer train already learns local placement ordering far better than SQI-SVM, and the pretrained/transfer transformer is much stronger still.

## Current Recommendation

Keep this branch. The benchmark is doing what we wanted:

1. SNR oracle fails, so the task is not a global-SNR shortcut.
2. SQI-SVM/MLP fail hard, especially on medium.
3. A short in-branch transformer already beats SQI-SVM/MLP and learns placement ordering.
4. A previous raw transformer transfers with a large margin over SQI baselines, so pretraining/continuation is the best route.

Next best step: continue from `e7_masked_pretrain_e6b_ft` or run a longer factorial fine-tune from that checkpoint. From-scratch 6-epoch training is not enough. The most important weakness to improve is held-out SNR-profile `8dB`, where short training is only `0.4203` and transfer is `0.5797`.

## Reproduce

```bash
python -m src.transformer_pipeline.run_factorial_local_all \
  --artifact_dir outputs/transformer_factorial_local \
  --source_artifact_dir outputs/transformer \
  --max_train_clean 120 \
  --max_val_clean 40 \
  --max_test_clean 40 \
  --force \
  --verbose

python -m src.transformer_pipeline.sqi_ml_multiclass \
  --transformer_artifact_dir outputs/transformer_factorial_local \
  --out_dir outputs/transformer_factorial_local_sqi_ml_three_class \
  --force \
  --verbose

python -m src.transformer_pipeline.evaluate_factorial_local \
  --artifact_dir outputs/transformer_factorial_local \
  --model_dir outputs/transformer_e6b_balanced_local/models/e7_masked_pretrain_e6b_ft \
  --sqi_summary outputs/transformer_factorial_local_sqi_ml_three_class/three_class_summary.json \
  --sqi_predictions outputs/transformer_factorial_local_sqi_ml_three_class/models/svm/svm_rbf_three_class_predictions_seed0.csv \
  --out_dir outputs/transformer_factorial_local/validation/e6b_transfer_eval

python -m src.transformer_pipeline.evaluate_factorial_local \
  --artifact_dir outputs/transformer_factorial_local \
  --model_dir outputs/transformer_factorial_local/models/factorial_local_t0_aux_short \
  --sqi_summary outputs/transformer_factorial_local_sqi_ml_three_class/three_class_summary.json \
  --sqi_predictions outputs/transformer_factorial_local_sqi_ml_three_class/models/svm/svm_rbf_three_class_predictions_seed0.csv \
  --out_dir outputs/transformer_factorial_local/validation/factorial_local_t0_aux_short
```
