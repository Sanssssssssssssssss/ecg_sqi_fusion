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

## External Full Reproduction

Full report reproduction is intentionally orchestrated from a sibling
workspace folder, not from a tracked `reproduce/` directory inside this repo:

```bash
python ../reproduce/run_reproduce.py --target baseline-cinc2011
python ../reproduce/run_reproduce.py --target baseline-but
python ../reproduce/run_reproduce.py --target conformer-cinc2011
python ../reproduce/run_reproduce.py --target conformer-but
```

That external controller fresh-clones this repository, installs dependencies in
an external virtual environment, and writes all generated data under
`../reproduce/work/`.

## Output Roots

```text
outputs/transformer/v116_e31/
outputs/transformer/supplemental/chapter4_evidence_frozen_final/
outputs/sqi_supplemental/
outputs/reports/
```

Top-level `report/` is reserved for the final submitted PDF/materials.

Supplemental targets are also split in the external controller as
`sqi-supplemental` and `transformer-supplemental`.
