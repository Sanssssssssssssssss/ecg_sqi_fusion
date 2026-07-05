# Reproducibility

## Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Use Python 3.11 where possible. `req.txt` omits `torch` for CUDA/cluster installs
where PyTorch is managed separately.

## Main Commands

```bash
python -m src.sqi_pipeline.run_all --verbose
python -m src.transformer_pipeline.run_all --run --train E31
python -m src.supplemental_sqi_experiments.run diagnose-existing
python -m src.supplemental_transformer_experiments.chapter4_evidence.run pipeline --run
```

## Required Checks

```bash
python -m compileall -q src/sqi_pipeline src/transformer_pipeline src/supplemental_sqi_experiments src/supplemental_transformer_experiments src/utils
python -m src.sqi_pipeline.cli --help
python -m src.transformer_pipeline.cli audit
python -m src.transformer_pipeline.cli train --model E31
python -m src.supplemental_transformer_experiments.chapter4_evidence.run pipeline
```

Expected Transformer audit:

```text
protocol rows: 31590 = 10530/10530/10530
train: 8310/8310/8310
val/test generated rows: 0
candidate types: original_but, but_native_morph, ptb_morph, clean_style
```

## Public-Data Smoke Scope

`python -m src.transformer_pipeline.cli clean-smoke --run` verifies that the
BUT/PTB public-data path can rebuild the required source assets. It is not an
exact replay of the frozen v116 support pool unless
`historical_support_exact=true` appears in `source/clean_smoke_summary.json`.
Fallback runs with `historical_support_exact=false` are valid engineering
smoke checks, not replacement Chapter 4 numbers.

## Output Roots

```text
outputs/transformer/v116_e31/
outputs/transformer/supplemental/chapter4_evidence_frozen_final/
outputs/sqi_supplemental/
outputs/reports/
```

Top-level `report/` is reserved for the final submitted PDF/materials.

For report-table and figure-source readiness checks, run:

```bash
python reproduce/check_reproduce.py artifact
```
