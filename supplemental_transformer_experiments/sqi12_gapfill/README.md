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
python -m supplemental_transformer_experiments.sqi12_gapfill.run pipeline --run --train
```

Useful single steps:

```bash
python -m supplemental_transformer_experiments.sqi12_gapfill.run manifest --run
python -m supplemental_transformer_experiments.sqi12_gapfill.run split --run
python -m supplemental_transformer_experiments.sqi12_gapfill.run build --run
python -m supplemental_transformer_experiments.sqi12_gapfill.run audit
python -m supplemental_transformer_experiments.sqi12_gapfill.run plot
python -m supplemental_transformer_experiments.sqi12_gapfill.run train --run
```
