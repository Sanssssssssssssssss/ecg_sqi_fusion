# Five-minute data-free inference

This tutorial verifies the repository without downloading a research dataset.
It uses one of the four frozen models and a NumPy ECG array.

## 1. Install

```bash
git clone https://github.com/Sanssssssssssssssss/ecg_sqi_fusion.git
cd ecg_sqi_fusion
python -m venv .venv
```

=== "Linux / macOS"

    ```bash
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

=== "Windows PowerShell"

    ```powershell
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```

Python 3.11 is the reference environment.

## 2. Verify the packaged models

```bash
python -m src.ecg_sqi_inference verify-bundles
```

The command checks every shipped model and profile against the SHA-256 values
in `pretrained/inference/manifest.json`.

## 3. Create a minimal input

```python
import numpy as np

fs = 125
t = np.arange(10 * fs) / fs
ecg = (0.1 * np.sin(2 * np.pi * 1.2 * t)).astype("float32")
np.save("example.npy", ecg)
```

This signal only checks the data path; it is not a clinically meaningful test
record.

## 4. Predict

```bash
python -m src.ecg_sqi_inference predict \
  --model singlelead-rbfsvm \
  --input example.npy \
  --fs 125 \
  --out example-output
```

The output directory contains:

- `example_segments.csv`: one prediction per complete 10-second segment;
- `all_segments.csv`: combined rows for every input record;
- `run_summary.json`: model, input, output, segment count, dropped tail, and errors.

Continue with [Inference and Docker](inference.md) for real inputs, lead-shape
rules, and all four models.
