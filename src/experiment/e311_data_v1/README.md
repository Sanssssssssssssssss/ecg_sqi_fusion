# Data v1 gap-fill line

This package freezes the current ECG SQI data-v1 gap-fill workflow in tracked
source. Large data bundles, generated arrays, checkpoints, and long reports stay
outside git.

```powershell
python -m src.experiment.e311_data_v1 audit
python -m src.experiment.e311_data_v1 plot
python -m src.experiment.e311_data_v1 build
python -m src.experiment.e311_data_v1 train-check --model both
```

Add `--run` to `build` or `train-check` only when you want to execute the
printed long-running commands.
