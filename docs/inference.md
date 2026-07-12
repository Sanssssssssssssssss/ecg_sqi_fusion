# Inference and Docker

The inference interface classifies non-overlapping 10-second ECG windows. It
does not train models or download datasets.

## Select a model

| Model | Leads | Classes |
|---|---:|---|
| `12lead-conformer` | 12 | acceptable / unacceptable |
| `12lead-rbfsvm` | 12 | acceptable / unacceptable |
| `singlelead-conformer` | 1 | good / medium / bad |
| `singlelead-rbfsvm` | 1 | good / medium / bad |

See the [model catalogue](models.md) for provenance and hashes.

## Local command

```bash
python -m src.ecg_sqi_inference predict \
  --model 12lead-conformer \
  --input /path/to/input \
  --fs 500 \
  --out /path/to/output \
  --device cpu
```

`--input` may be one file or a recursively scanned directory. Supported inputs
are `.npy`, `.npz`, numeric `.csv`, and WFDB `.hea` records. The source
sampling frequency is supplied once with `--fs`; data are resampled to 125 Hz.

## Shape contract

- Single-lead: `(samples,)`, `(samples, 1)`, or `(1, samples)`.
- Twelve-lead: `(samples, 12)` or `(12, samples)`.
- At least 1,250 resampled samples are required.
- An incomplete final window is reported as `dropped_seconds` and not padded.

## Docker

Build once from the repository root:

```bash
docker build -f docker/inference/Dockerfile -t ecg-sqi-infer .
docker run --rm ecg-sqi-infer verify-bundles
```

Mount the same directory for input and output:

```bash
docker run --rm -v /host/ecg:/data ecg-sqi-infer predict \
  --model singlelead-conformer \
  --input /data/input \
  --fs 500 \
  --out /data/output
```

For WSL, Windows drives appear below `/mnt`; for example,
`E:\ecg-data` becomes `/mnt/e/ecg-data`.

## Python API

```python
from pathlib import Path

from src.ecg_sqi_inference.core import predict_records
from src.ecg_sqi_inference.models import get_predictor

summary = predict_records(
    input_path=Path("example.npy"),
    out_dir=Path("example-output"),
    fs=125,
    predictor=get_predictor("singlelead-rbfsvm"),
)
```

The stable functions and error contracts are documented in the
[Python API](api_reference.md).

