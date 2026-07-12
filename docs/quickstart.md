# Quick Start

Use Python 3.11 where possible.

```bash
git clone https://github.com/Sanssssssssssssssss/ecg_sqi_fusion.git
cd ecg_sqi_fusion
python -m venv .venv
pip install -r requirements.txt
```

Run the two main research workflows:

```bash
python -m src.sqi_pipeline.run_all --verbose
python -m src.transformer_pipeline.run_all --run --train E31
```

Run the checks:

```bash
python -m pytest -q
python -m src.ecg_sqi_inference verify-bundles
```

For a data-free first run, use the [Docker inference image](inference.md).

