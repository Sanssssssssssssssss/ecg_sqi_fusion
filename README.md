# ECG SQI Fusion

Research code for ECG signal-quality assessment with classical SQI baselines
and waveform Conformer models. Raw data is downloaded from public sources on
demand; generated artifacts and curated evidence live under `outputs/`.

## Quick Start

```bash
pip install -r requirements.txt

python -m src.sqi_pipeline.run_all --verbose
python -m src.transformer_pipeline.run_all --run --train E31
python -m src.supplemental_sqi_experiments.run diagnose-existing
python -m src.supplemental_transformer_experiments.chapter4_evidence.run pipeline --run
```

Use Python 3.11 where possible. If PyTorch is installed separately, `req.txt`
contains the lighter dependency list without `torch`.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/sqi_pipeline/` | Classical SQI feature, SVM, and LM-MLP baseline pipeline. |
| `src/transformer_pipeline/` | Official BUT v116 gap-fill data line and E31 Conformer. |
| `src/supplemental_sqi_experiments/` | SQI supplemental audits and reproduction checks. |
| `src/supplemental_transformer_experiments/` | Transformer supplemental and Chapter 4 evidence runs. |
| `src/utils/` | Shared paths, downloads, and small reporting helpers. |
| `outputs/` | Generated artifacts. Curated frozen Chapter 4 evidence is tracked. |
| `docs/` | Archived v116 method notes and small reference figures. |
| `report/` | Final submission PDF/materials only. |

The old top-level `reports/` path is intentionally ignored and unused.

## Official Transformer Line

Current mainline:

```text
BUT gap5 originals -> record-heldout split -> train-only medium/bad gap fill
-> waveform-derived channels -> E31 query-mean fused Conformer
```

Audit contract:

```text
policy: v116_gapfill_dual_goodorig_nm40_ms10_smc_s20260876
protocol: 31590 = 10530/10530/10530
train: 8310/8310/8310
val/test: original_but only
sampler: raw rows, no record-balanced sampler
```

Checked snapshot, seed `20260876`:

```text
E31 test acc: 0.9432
E31 macro F1: 0.9525
good/medium/bad recall: 0.9421 / 0.9320 / 0.9939
```

Useful checks:

```bash
python -m src.transformer_pipeline.cli audit
python -m src.transformer_pipeline.cli train --model E31
python -m compileall -q src/sqi_pipeline src/transformer_pipeline src/supplemental_sqi_experiments src/supplemental_transformer_experiments src/utils
```

## Data And Outputs

Pipeline entrypoints check for required local data and download missing public
WFDB databases into `data/`. Raw datasets, checkpoints, and regenerated arrays
are not committed.

Public-data rebuilds are engineering checks for the pipeline. Exact replay of
the tracked frozen v116 evidence depends on the support assets that were present
when that evidence package was frozen; clean public-data rebuilds should be
reported separately from the frozen paper numbers.

Important generated locations:

```text
outputs/transformer/v116_e31/
outputs/transformer/supplemental/chapter4_evidence_frozen_final/
outputs/sqi_supplemental/
outputs/reports/
```

See `DATA_AVAILABILITY.md` and `REPRODUCIBILITY.md` for the compact release
notes.

Full report reproduction is orchestrated outside this repository from the
workspace-level `../reproduce/` controller so fresh clones, temporary data, and
generated outputs stay isolated from the main project tree.

Paper-aligned SQI targets use `wfdb-qrs-kit` to manage the external
`wqrs` and EP Limited/Hamilton detector executables; detector provenance is
written by `python -m src.sqi_pipeline.qrs.setup_paper_detectors`.
The Python wrapper is a project dependency, but local detector binaries or
sources remain under their upstream WFDB/EP Limited GPL/LGPL terms and are not
relicensed by this repository.
