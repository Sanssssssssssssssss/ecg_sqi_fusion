# ECG SQI Fusion

Research code for ECG signal-quality assessment and noise-robust modelling. The repository currently contains two related workflows:

1. A classical SQI pipeline built around PhysioNet Challenge 2011 `set-a` and NSTDB.
2. A PTB-XL pipeline for noisy Lead I segment generation, multi-task transformer training, and evaluation.

## Repository layout

`src/sqi_pipeline/`
Classical SQI/ML pipeline package and command line entrypoint.

`src/transformer_pipeline/`
PTB-XL Lead I transformer data, training, evaluation, and diagnostics pipeline.

`src/utils/`
Shared project-root helpers.

`slurm/run_ampere.sh`
Example SLURM job for training on Cambridge CSD3 Ampere GPUs.

## Environment

Use Python 3.11 if possible.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are installing PyTorch separately for a cluster or CUDA-specific setup, `req.txt` is a lighter dependency list without `torch`.

## Data layout

The code assumes the project root can see the following folders:

```text
data/
  physionet/
    challenge-2011/
      set-a/
    nstdb/
  ptb-xl/
artifacts/
artifact1/
```

The SQI pipeline writes to `artifacts/`. The transformer pipeline writes to `artifact1/`.

## Classical SQI pipeline

The default runner is:

```bash
python -m src.sqi_pipeline.cli --verbose
```

`src.sqi_pipeline.cli` executes:

- raw manifest creation
- set-a split generation
- balanced noise synthesis
- resampling to 125 Hz
- QRS cache generation
- 84-feature extraction
- feature normalisation
- baseline model training

Useful flags:

```bash
python -m src.sqi_pipeline.cli --fresh
python -m src.sqi_pipeline.cli --only manifest_raw,split_seta,record84
python -m src.sqi_pipeline.cli --force
python -m src.sqi_pipeline.validate_outputs --write artifacts/validation/current_seed0.json
```

## PTB-XL workflow

The default transformer runner is:

```bash
python -m src.transformer_pipeline.cli --verbose
```

It executes:

- filter PTB-XL metadata to exclude Lead I noise labels
- build the Lead I manifest
- make 10 s, 125 Hz Lead I segments
- split clean segments by `ecg_id`
- synthesize balanced SNR classes with NSTDB noise
- generate RR-level pseudo noise labels
- run a model forward check
- train the multi-task transformer
- evaluate the best checkpoint

Useful commands:

```bash
python -m src.transformer_pipeline.cli --only train --dry-run --verbose
python -m src.transformer_pipeline.cli --only train --verbose
python -m src.transformer_pipeline.cli --only evaluate --verbose
python -m src.transformer_pipeline.validate_outputs --write artifact1/validation/current_seed0.json
```

Cluster training:

```bash
sbatch slurm/run_ampere.sh
```

The transformer train step expects prepared arrays under `artifact1/datasets/` and writes checkpoints and reports under `artifact1/models/mtl_transformer_seed0_step6/`.

## Cluster usage

For CSD3 Ampere:

```bash
sbatch slurm/run_ampere.sh
```

The SLURM script assumes the repository lives at `/home/cx272/final_project/ecg_sqi_fusion` and that a local virtual environment already exists at `.venv/`.

## Notes

- This is research code with fixed model hyperparameters inside the training module.
- `src/utils/paths.py` resolves the project root automatically from `.git` or `pyproject.toml`.
