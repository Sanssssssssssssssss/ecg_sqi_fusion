# ECG SQI Fusion

Research code for ECG signal-quality assessment and noise-robust modelling. The repository currently contains two related workflows:

1. A classical SQI pipeline built around PhysioNet Challenge 2011 `set-a` and NSTDB.
2. A PTB-XL pipeline for noisy Lead I segment generation, multi-task transformer training, and evaluation.

## Repository layout

`src/sqi_pipeline/`
Classical SQI/ML pipeline package and command line entrypoint.

`src/data/`
PTB-XL preprocessing helpers.

`src/preprocess/`
PTB-XL preprocessing scripts.

`src/models/`
PTB-XL multi-task transformer code.

`src/experiment/`
PTB-XL evaluation, debugging, and visualisation scripts.

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

The classical pipeline mainly writes to `artifacts/`. The PTB-XL scripts currently use `artifact1/` for manifests, datasets, checkpoints, and reports.

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

The PTB-XL work is script-driven and separate from the SQI package. The main scripts are:

```text
src/data/filter_ptbxl_lead_i.py
src/data/ptbxl_step1_make_manifest_leadI.py
src/data/ptbxl_step3_split_clean_segments.py
src/noise/ptbxl_step4_synthesize_noise_snr_balanced.py
src/noise/ptbxl_step5_make_rr_pseudo_noise_level.py
src/models/ptbxl_step6_train_mtl_transformer.py
src/experiment/ptbxl_eval_best_step6.py
```

Training:

```bash
python -u src/models/ptbxl_step6_train_mtl_transformer.py
```

Evaluation of the best saved checkpoint:

```bash
python -u src/experiment/ptbxl_eval_best_step6.py
```

The current transformer script expects prepared PTB-XL arrays under `artifact1/datasets/` and writes checkpoints and reports under `artifact1/models/mtl_transformer_seed0_step6/`.

## Cluster usage

For CSD3 Ampere:

```bash
sbatch slurm/run_ampere.sh
```

The SLURM script assumes the repository lives at `/home/cx272/final_project/ecg_sqi_fusion` and that a local virtual environment already exists at `.venv/`.

## Notes

- This is research code with several fixed paths and hyperparameters inside scripts.
- `src/utils/paths.py` resolves the project root automatically from `.git` or `pyproject.toml`.
- The PTB-XL workflow is still under active development, so intermediate experiment scripts under `src/experiment/` are intentionally kept in the repository.
