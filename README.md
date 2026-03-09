# ECG SQI Fusion

Research code for ECG signal-quality assessment and noise-robust modelling. The repository currently contains two related workflows:

1. A classical SQI pipeline built around PhysioNet Challenge 2011 `set-a` and NSTDB.
2. A PTB-XL pipeline for noisy Lead I segment generation, multi-task transformer training, and evaluation.

## Repository layout

`src/run_all.py`
Main entrypoint for the classical end-to-end pipeline.

`src/data/`
Manifest creation, train/validation/test splitting, and PTB-XL preprocessing helpers.

`src/preprocess/`
Signal resampling and preprocessing utilities.

`src/qrs/`
QRS detector wrappers and cached R-peak generation.

`src/features/`
SQI feature extraction, including the 84-feature record representation.

`src/models/`
Baseline models and the PTB-XL multi-task transformer.

`src/experiment/`
Evaluation, debugging, and visualisation scripts.

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
python src/run_all.py --verbose
```

`src/run_all.py` is configured to execute:

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
python src/run_all.py --fresh
python src/run_all.py --only manifest_raw,split_seta,record84
python src/run_all.py --force
```

## PTB-XL workflow

The PTB-XL work is script-driven rather than wired into `run_all.py`. The main scripts are:

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
