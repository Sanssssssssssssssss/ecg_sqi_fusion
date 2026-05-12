# Transformer Local-Aware Tuning Report

Date: 2026-05-12

## Summary

Two tuning jobs were run after the local-aware architecture changes:

- `e3_pos_topk` on the E3 triplet/global-SNR benchmark
- `e6b_bound` on the E6b balanced local-counterfactual benchmark

The 0.945 target was not reached as a test result. The closest old number around 0.945 was the original tune09 validation accuracy, not test accuracy. The strongest previous single-model test result remains E3 triplet `0.9405`.

## Runs

| run | artifact | job | result |
|---|---|---:|---|
| E3 baseline | `outputs/transformer_e3_triplet_k1/models/e3_triplet_tune09` | old | best test on global/E3 |
| E3 pos/topk | `outputs/transformer_e3_triplet_k1/models/e3_pos_meanmax_topk` | 29242333 | worse than E3 baseline |
| E6b layer2 | `outputs/transformer_e6b_balanced_local/models/e6b_arch2_pos_localpool` | old | previous best local-aware E6b |
| E6b boundary | `outputs/transformer_e6b_balanced_local/models/e6b_arch2_boundary_tune` | 29242334 | small E6b improvement |

## E3 / Global Benchmark

| model | test acc | balanced acc | macro F1 | good recall | medium recall | bad recall | best val |
|---|---:|---:|---:|---:|---:|---:|---:|
| tune09 original | 0.9356 | 0.9356 | 0.9354 | 0.9612 | 0.8898 | 0.9559 | 0.9457 |
| E3 triplet baseline | 0.9405 | 0.9405 | 0.9406 | 0.9622 | 0.9192 | 0.9401 | 0.9590 |
| E3 pos/topk | 0.9335 | 0.9335 | 0.9334 | 0.9685 | 0.8940 | 0.9380 | 0.9562 |

E3 pos/topk confusion matrix:

```text
[[923,  28,   2],
 [ 73, 852,  28],
 [ 14,  45, 893]]
```

Interpretation:

- The new positional + mean/max/top-k pooling did not improve E3.
- Best validation was epoch 1, then training overfit and later epochs became unstable.
- Test accuracy dropped from `0.9405` to `0.9335`; medium recall dropped from `0.9192` to `0.8940`.
- This does not justify replacing the E3 baseline architecture.

## E6b / Local Benchmark

E6b is not comparable to the old 0.945 target because it intentionally breaks the global-SNR shortcut and makes local noise placement matter.

| model | test acc | balanced acc | macro F1 | good recall | medium recall | bad recall | best val |
|---|---:|---:|---:|---:|---:|---:|---:|
| E6b baseline | 0.7906 | 0.7799 | 0.7785 | 0.7904 | 0.6108 | 0.9386 | 0.7988 |
| E6b layer2 | 0.7960 | 0.7863 | 0.7862 | 0.8062 | 0.6359 | 0.9168 | 0.8010 |
| E6b boundary | 0.7992 | 0.7912 | 0.7896 | 0.7729 | 0.6569 | 0.9437 | 0.8016 |

E6b boundary confusion matrix:

```text
[[1412, 307, 108],
 [ 345, 940, 146],
 [  16,  82, 1644]]
```

Interpretation:

- Boundary tuning gives the best E6b test so far: `0.7992`.
- Medium recall improves from layer2 `0.6359` to `0.6569`.
- Bad recall recovers from `0.9168` to `0.9437`.
- Good recall drops from `0.8062` to `0.7729`.
- This is a real but small local-benchmark gain, not a 0.945-style breakthrough.

## Recommendation

For the old/global benchmark, do not keep `mean_max_topk` + positional embedding as the recommended model. It underperforms the existing E3 baseline.

For E6b, keep the result as an experiment because it is currently the best local benchmark number, but keep it isolated. I would not make ordinal/SNR/local-aware boundary tuning the default mainline yet.

Rollback guidance:

- If the goal is the cleanest stable transformer line, revert the local-aware architecture code commit `b6c0124`.
- If the goal is to preserve E6b research evidence, keep the code as optional flags only and keep default behavior at `cls_pooling="mean"` with no positional embedding.
- Keep the E6b dataset and reports; the benchmark itself is useful even if the model changes are later rolled back.

## Artifacts

Important outputs:

- `outputs/transformer_e3_triplet_k1/models/e3_pos_meanmax_topk/test_report.json`
- `outputs/transformer_e3_triplet_k1/models/e3_pos_meanmax_topk/probe_summary.json`
- `outputs/transformer_e6b_balanced_local/models/e6b_arch2_boundary_tune/test_report.json`
- `outputs/transformer_e6b_balanced_local/models/e6b_arch2_boundary_tune/probe_summary.json`

