# Chapter 4 Raw Evidence

Independent raw-evidence runner for Chapter 4 experiments. It does not modify
the official Transformer or SQI pipelines.

Output root:

```powershell
outputs/transformer/supplemental/chapter4_evidence_frozen_final/
```

Full run:

```powershell
python -m src.supplemental_transformer_experiments.chapter4_evidence.run pipeline --run
```

Single steps:

```powershell
python -m src.supplemental_transformer_experiments.chapter4_evidence.run seta-build --run
python -m src.supplemental_transformer_experiments.chapter4_evidence.run seta-sqi --run
python -m src.supplemental_transformer_experiments.chapter4_evidence.run audit --run
python -m src.supplemental_transformer_experiments.chapter4_evidence.run seta-repair --run
python -m src.supplemental_transformer_experiments.chapter4_evidence.run seta-models --run
python -m src.supplemental_transformer_experiments.chapter4_evidence.run but-models --run
python -m src.supplemental_transformer_experiments.chapter4_evidence.run figures --run
python -m src.supplemental_transformer_experiments.chapter4_evidence.run report --run
```

Figures use Python/matplotlib only and export PNG, SVG, PDF, and TIFF with
source-data CSVs.

The frozen-final root is the current report source. Older
`chapter4_evidence/` outputs are stale and should not be cited.
