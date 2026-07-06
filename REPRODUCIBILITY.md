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

## Full Report Reproduction

Report reproduction is split by model family and dataset. Each command writes
only under `reproduce/work/` and emits `summary.json`, `summary.md`,
`audit_matrix.csv`, and `artifact_checksums.csv`.

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File reproduce/scripts/ps1/baseline_cinc2011.ps1
powershell -ExecutionPolicy Bypass -File reproduce/scripts/ps1/baseline_but.ps1
powershell -ExecutionPolicy Bypass -File reproduce/scripts/ps1/conformer_cinc2011.ps1
powershell -ExecutionPolicy Bypass -File reproduce/scripts/ps1/conformer_but.ps1
```

Linux, macOS, or WSL:

```bash
bash reproduce/scripts/sh/baseline_cinc2011.sh
bash reproduce/scripts/sh/baseline_but.sh
bash reproduce/scripts/sh/conformer_cinc2011.sh
bash reproduce/scripts/sh/conformer_but.sh
```

Supplemental:

```bash
bash reproduce/scripts/sh/supplemental/sqi_supplemental.sh
bash reproduce/scripts/sh/supplemental/transformer_supplemental.sh
```

The clean public rebuild path is intended to be runnable from a fresh clone.
It is not claimed to be an exact byte replay of the frozen v116 support pool.

## Output Roots

```text
outputs/transformer/v116_e31/
outputs/transformer/supplemental/chapter4_evidence_frozen_final/
outputs/sqi_supplemental/
outputs/reports/
```

Top-level `report/` is reserved for the final submitted PDF/materials.

For direct Python use, run `python reproduce/run_reproduce.py --target <target>`.
