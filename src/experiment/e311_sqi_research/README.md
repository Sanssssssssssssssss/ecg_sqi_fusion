# E3.11 SQI Research Experiments

This package is intentionally isolated from `src/sqi_pipeline` and
`src/transformer_pipeline`. It reads the current E3.11f data and D1 warm-start
checkpoint, then writes all experiment outputs under:

```text
outputs/experiment/e311_sqi_research/
```

The first screening pass uses seed 1. Runs are only worth expanding when they
beat the local-mask candidate (`0.9505`), improve medium recall without hurting
bad recall, or show lower multi-task conflict while staying near the strong
baseline (`0.9464`).

## Commands

List recipes:

```bash
python -m src.experiment.e311_sqi_research.train --list
```

Dry-run one recipe:

```bash
python -m src.experiment.e311_sqi_research.train \
  --group head_reimpl \
  --task_id 0 \
  --dry_run
```

Summarize available results:

```bash
python -m src.experiment.e311_sqi_research.summarize
```

Submit screening arrays:

```bash
sbatch src/experiment/e311_sqi_research/slurm/run_loss_conflict.sh
sbatch src/experiment/e311_sqi_research/slurm/run_head_reimpl.sh
sbatch src/experiment/e311_sqi_research/slurm/run_target_gate_reimpl.sh
sbatch src/experiment/e311_sqi_research/slurm/run_generalization_loss.sh
```
