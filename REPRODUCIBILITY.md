# Reproducibility

## Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Use Python 3.11 where possible. `req.txt` omits `torch` for CUDA/cluster installs
where PyTorch is managed separately.

Paper-aligned SQI runs require the external `wqrs` and EP Limited/Hamilton
detectors. The Python wrapper/setup layer is `wfdb-qrs-kit`; install or copy the
detectors before running `paper_aligned`:

```bash
python -m src.sqi_pipeline.qrs.setup_paper_detectors --out_dir outputs/sqi_paper_aligned/qrs/tools --from-bin-dir <wfdb-qrs-bin> --no_download --require
```

If a WFDB C development environment is available, use `--compile` instead of
`--from-bin-dir`. The setup command writes detector provenance to
`paper_qrs_detector_manifest.json`. The Python wrapper is installed as
`wfdb-qrs-kit`, but copied or compiled `wqrs` and EP Limited/Hamilton detector
binaries/sources remain under their upstream WFDB/EP Limited GPL/LGPL terms;
this repository does not relicense those local detector artifacts.

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
python ../reproduce/run_reproduce.py --target inference-service
```

That external controller fresh-clones this repository, installs dependencies in
an external virtual environment, and writes all generated data under
`../reproduce/work/`.
For paper SQI targets, set `WFDB_QRS_KIT_FROM_BIN_DIR` or the
`WFDB_QRS_KIT_*_EXE` variables before launching the controller, unless the
detectors are already on `PATH` or installed in the `wfdb-qrs-kit` cache.

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

The `inference-service` target is data-free: it installs a fresh clone and
verifies the hashes of all model files shipped in the inference container.
