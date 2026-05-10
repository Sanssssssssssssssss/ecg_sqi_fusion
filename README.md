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
outputs/sqi/
outputs/transformer/
```

The SQI pipeline writes to `outputs/sqi/`. The transformer pipeline writes to `outputs/transformer/`.

## Classical SQI pipeline

Run the full classical SQI line:

```bash
python -m src.sqi_pipeline.run_all --verbose
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
python -m src.sqi_pipeline.validate_outputs --write outputs/sqi/validation/current_seed0.json
```

## PTB-XL workflow

Transformer preprocessing and transformer training are intentionally separate.

Run all preprocessing/data steps:

```bash
python -m src.transformer_pipeline.run_preprocess_all --verbose
```

It executes:

- filter PTB-XL metadata to exclude Lead I noise labels
- build the Lead I manifest
- make 10 s, 125 Hz Lead I segments
- split clean segments by `ecg_id`
- synthesize balanced SNR classes with NSTDB noise
- generate RR-level pseudo noise labels

Run the transformer/model steps:

```bash
python -m src.transformer_pipeline.run_transformer_all --verbose
```

It executes:

- run a model forward check
- train the multi-task transformer
- evaluate the best checkpoint

Useful commands:

```bash
python -m src.transformer_pipeline.run_preprocess_all --only segments --verbose
python -m src.transformer_pipeline.run_transformer_all --dry-run --verbose
python -m src.transformer_pipeline.run_transformer_all --only train --verbose
python -m src.transformer_pipeline.run_transformer_all --only evaluate --verbose
python -m src.transformer_pipeline.validate_outputs --write outputs/transformer/validation/current_seed0.json
```

Single-step scripts can also be run directly, for example:

```bash
python -m src.transformer_pipeline.data.filter_lead_i --verbose
python -m src.transformer_pipeline.data.make_manifest_lead_i --verbose
python -m src.transformer_pipeline.preprocess.make_segments_10s_125hz --verbose
python -m src.transformer_pipeline.data.make_clean_split --verbose
python -m src.transformer_pipeline.noise.synthesize_snr_dataset --verbose
python -m src.transformer_pipeline.noise.make_rr_noise_level --verbose
python -m src.transformer_pipeline.train --verbose
python -m src.transformer_pipeline.evaluate --verbose
```

Cluster training:

```bash
sbatch slurm/run_ampere.sh
```

The transformer train step expects prepared arrays under `outputs/transformer/datasets/` and writes checkpoints and reports under `outputs/transformer/models/mtl_transformer_seed0_step6/`.

## Cluster usage

For CSD3 Ampere:

```bash
sbatch slurm/run_ampere.sh
```

The SLURM script assumes the repository lives at `/home/cx272/final_project/ecg_sqi_fusion` and that a local virtual environment already exists at `.venv/`.

## Notes

- This is research code with fixed model hyperparameters inside the training module.
- `src/utils/paths.py` resolves the project root automatically from `.git` or `pyproject.toml`.
