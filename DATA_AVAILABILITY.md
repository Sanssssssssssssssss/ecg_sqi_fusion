# Data Availability

This repository does not redistribute raw ECG databases.

The pipelines download public WFDB data into `data/` when required:

| Dataset | Use |
|---|---|
| PhysioNet/CinC Challenge 2011 Set-A | Classical SQI and 12-lead supplemental experiments. |
| MIT-BIH Noise Stress Test Database | EM/MA noise controls for SQI experiments. |
| PTB-XL | Clean ECG carriers for gap-fill proposals. |
| BUT QDB | Official single-lead Transformer v116 experiment. |

Generated arrays, tables, figures, predictions, and local raw data stay outside
version control and are recreated under `outputs/`. Only the four inference
model artifacts and their runtime profiles are tracked. Public-data rebuilds
can validate the pipeline, but exact frozen v116 support-pool replay requires
the archived support assets used when that evidence was frozen; otherwise the
clean-smoke report marks `historical_support_exact=false`.
