# Supplemental Transformer Experiments

Official v116/E31 runs live in `src/transformer_pipeline/` and write to
`outputs/transformer/v116_e31/`.

This folder is only for lightweight reproduction wrappers and extra diagnostics.
Generated reports, plots, checkpoints, and sample galleries must go under
`outputs/transformer/supplemental/`.

Useful commands:

```bash
python -m src.transformer_pipeline.cli audit
python -m src.transformer_pipeline.cli plot
python -m src.transformer_pipeline.cli train --model E31
```
