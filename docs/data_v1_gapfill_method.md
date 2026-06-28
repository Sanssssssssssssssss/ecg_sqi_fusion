# Data v1 Gap-Fill Method

This document freezes the current reproducible data line tagged as
`data-v1.0`.

## Method Summary

Data v1 starts from the BUT gap5 extracted 10 s windows and balances only the
training split. Validation and test remain native BUT originals.

Fixed protocol:

```text
policy: v116_gapfill_dual_goodorig_nm99_ms10_rnd_s20260876
seed: 20260876
final protocol rows: 31590
final class rows: good 10530, medium 10530, bad 10530
```

Original BUT gap5 rows:

```text
total  18635
good   10530
medium  6449
bad     1656
```

Original split before train balancing:

```text
train original: good 8424, medium 5199, bad 1314
val original:   good 1053, medium  618, bad  178
test original:  good 1053, medium  632, bad  164
```

Final training split after exact train-only balance:

```text
train final: good 8310, medium 8310, bad 8310
val final:   1849 rows, original_but only
test final:  1849 rows, original_but only
```

The 114 surplus good train originals and surplus train-linked generated rows are
marked `unused`; they are not deleted from the protocol bundle.

## Generation Rules

Candidate labels are part of the public protocol contract:

```text
original_but
but_native_morph
ptb_morph
clean_style
```

Rules:

- `good` is not generated for the final balanced protocol.
- `medium` and `bad` keep all eligible original BUT rows, then fill only the
  training gap.
- `but_native_morph` uses BUT carriers with small morphology/acquisition
  perturbations: shift, gain, baseline drift, noise, and class-specific light
  artifact mixing.
- `ptb_morph` uses PTB carriers aligned to BUT style/residual support.
- `clean_style` is capped to a tiny supplement and is not a main component.
- Generated rows are allowed in train only when their donor/linkage resolves to
  original train support.

Current exact train composition:

```text
good:
  original_but 8310

medium:
  original_but      5199
  but_native_morph  3050
  clean_style         30
  ptb_morph           31

bad:
  original_but      1314
  but_native_morph  6867
  clean_style         62
  ptb_morph           67
```

## Audits

Main data-side acceptance is dual-view waveform generated-vs-original
separability, evaluated only on `medium` and `bad`; `good` is excluded because
it has no generated rows by design.

```text
medium sym AUC 0.510
bad    sym AUC 0.529
pooled sym AUC 0.549
```

Leakage and integrity checks:

```text
val/test generated rows: 0
train generated donor-split problems: 0
allowed candidate types:
  original_but, but_native_morph, ptb_morph, clean_style
missing class rows: 0
missing signal idx rows: 0
```

## Model Check

Training uses raw rows, not the record-balanced sampler.

Model input remains:

```text
signals.npz -> dual-view waveform channels
```

No SQI table columns are concatenated to the model input. Factor/SQI-like targets
remain auxiliary supervision, so the model can learn SQI-like waveform
representations but does not receive SQI features as inputs.

Exact-balanced checks:

```text
E4:
  val  acc 0.9340, macro F1 0.9385
  test acc 0.9389, macro F1 0.9431
  test recalls: good 0.9687, medium 0.8813, bad 0.9695

E24:
  val  acc 0.9319, macro F1 0.9363
  test acc 0.9362, macro F1 0.9413
  test recalls: good 0.9592, medium 0.8877, bad 0.9756
```

## Reproduction Commands

Audit the current materialized protocol:

```powershell
python -m src.transformer_pipeline.data_v1_gapfill audit
```

Create the small method figures:

```powershell
python -m src.transformer_pipeline.data_v1_gapfill plot
```

Print the exact protocol build command:

```powershell
python -m src.transformer_pipeline.data_v1_gapfill build
```

Run the exact protocol build command:

```powershell
python -m src.transformer_pipeline.data_v1_gapfill build --run
```

Print the exact split command:

```powershell
python -m src.transformer_pipeline.data_v1_gapfill split
```

Run the exact split command:

```powershell
python -m src.transformer_pipeline.data_v1_gapfill split --run
```

Print the exact E4/E24 training check commands:

```powershell
python -m src.transformer_pipeline.data_v1_gapfill train-check --model both
```

Run the exact E4/E24 training checks:

```powershell
python -m src.transformer_pipeline.data_v1_gapfill train-check --model both --run
```

Run the full line:

```powershell
python -m src.transformer_pipeline.data_v1_gapfill pipeline --run --train both
```

## Figures

![Original BUT class counts](data_v1_figures/original_but_class_counts.png)

![Split class counts](data_v1_figures/split_class_counts.png)

![Train candidate-type composition](data_v1_figures/train_candidate_type_composition.png)

![Dual AUC summary](data_v1_figures/dual_auc_summary.png)

![Model check accuracy](data_v1_figures/model_check_accuracy.png)
