# SQI Set-A 12-Lead Gap-Fill

Isolated supplemental experiment for applying the v116 idea to the SQI
baseline Set-A dataset. It does not modify the official Transformer or SQI
pipelines.

Outputs go to:

```text
outputs/transformer/supplemental/sqi12_gapfill/
```

Run the full experiment:

```bash
python -m src.supplemental_transformer_experiments.sqi12_gapfill.run pipeline --run --train
```

Useful single steps:

```bash
python -m src.supplemental_transformer_experiments.sqi12_gapfill.run manifest --run
python -m src.supplemental_transformer_experiments.sqi12_gapfill.run split --run
python -m src.supplemental_transformer_experiments.sqi12_gapfill.run build --run
python -m src.supplemental_transformer_experiments.sqi12_gapfill.run audit
python -m src.supplemental_transformer_experiments.sqi12_gapfill.run plot
python -m src.supplemental_transformer_experiments.sqi12_gapfill.run train --run
python -m src.supplemental_transformer_experiments.sqi12_gapfill.run sqi-baselines --run
```

Training defaults use batch size 24 on 8 GB GPUs; use `--grad-accum-steps` to
try a larger effective batch without changing the model.

Current waveform-only E31-style default is lead-wise shared: each lead becomes
8 waveform-derived channels, the same Conformer encodes all 12 leads, and
mean+max pooling makes the record decision. No SQI scalar input is used.

Current generated unacceptable gap fill is PTB-heavy by design:

```text
ptb12_morph 211 / 383 = 55.1%
seta_native_morph 153 / 383 = 39.9%
noise_style 19 / 383 = 5.0%
```

The record head still uses the original E31-style query evidence. No guard,
SQI scalar input, or extra classifier head is added for this supplemental line.
The low-rate `noise_style` fallback uses small random NSTDB `em/ma` noise when
available, not synthetic white Gaussian noise.
