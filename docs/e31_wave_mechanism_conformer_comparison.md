# E31 Wave-Mechanism Conformer Comparison

> Archive note: this comparison is a frozen checkpoint-level snapshot. The
> current official pipeline command and data contract live in `README.md`;
> generated Chapter 4 evidence tables live under
> `outputs/transformer/supplemental/`.

## Summary

`E31_wave_mechanism_conformer` is the main model for the v116 gap-fill data
line. It keeps the E4 waveform backbone and the E24 mechanism auxiliary
contract, but removes the E24 good/medium-specific decision shortcuts.

Current data policy:

```text
v116_gapfill_dual_goodorig_nm40_ms10_smc_s20260876
train: 8310/8310/8310, balanced with generated medium/bad only
val/test: original_but only
sampler: raw rows, no record-balanced sampler
```

## Model Contract

E31 reuses the existing GM mechanism runner and model class. No new backbone was
added.

```text
factor_contract = mechanism
gm_mode = direct
factor_weight = 0.16
local_weight = 0.14
artifact_weight = 0.14
subtype_weight = 0.02
subtype_class_consistency_weight = 0.0
subtype_class_fusion_alpha = 0.0
use_pairrank = False
pairrank_weight = 0.0
hard_gm_weight = 0.0
medium_bad_guard_weight = 0.0
class_weights = [1.0, 1.08, 1.08]
```

Interpretation: local maps and factor predictions remain auxiliary evidence.
They do not directly correct the good/medium logit, and there is no pairrank,
subtype-class fusion, or guard loss in the main decision path.

## Commands

```powershell
python -m src.transformer_pipeline.data_v1_gapfill train-check --model all --run
```

Dry-run one candidate:

```powershell
python -m src.transformer_pipeline.data_v1_gapfill train-check --model E31
```

## Clean Test Metrics

All metrics below are recomputed from saved `test_predictions.npz` files with
the same argmax/probability contract.

| model | acc | macro F1 | good recall | medium recall | bad recall | bad FPR nonbad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E4 query high-res local-art | 0.946998 | 0.956902 | 0.955366 | 0.920886 | 0.993902 | 0.000593 |
| E24 subtype-fusion pairrank | 0.924283 | 0.934220 | 0.932574 | 0.892405 | 0.993902 | 0.004154 |
| E31 wave-mechanism conformer | 0.946998 | 0.957427 | 0.935423 | 0.954114 | 0.993902 | 0.000593 |

Confusion matrices use rows=true labels and columns=predicted labels in
`good, medium, bad` order.

```text
E4:
[[1006, 47, 0],
 [  49, 582, 1],
 [   0, 1, 163]]

E24:
[[982, 71, 0],
 [ 61, 564, 7],
 [  0, 1, 163]]

E31:
[[985, 68, 0],
 [ 28, 603, 1],
 [  0, 1, 163]]
```

## Readout

E31 matches E4 test accuracy and is slightly better on macro F1. Its main gain
is the good/medium boundary: medium recall rises from E4's `0.920886` to
`0.954114`, while good recall drops from `0.955366` to `0.935423`.

E24 is weaker in this run, mainly because medium recall falls to `0.892405`.
That supports using E24's factor-fused GM head, pairrank, and subtype fusion as
ablations rather than the main manuscript model.

![E31 model comparison](data_v1_figures/e31_model_comparison.png)

## Files

- Candidate config: `src/transformer_pipeline/data_v1_gapfill/support/run_gm_mechanism_repair_suite.py`
- CLI: `src/transformer_pipeline/data_v1_gapfill/train_check.py`
- Figure: `docs/data_v1_figures/e31_model_comparison.png`
