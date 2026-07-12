# ECG SQI inference image

The image runs the four frozen inference models only; it does not train models or download data.

## Build and verify

From the repository root:

```bash
docker build -f docker/inference/Dockerfile -t ecg-sqi-infer .
docker run --rm ecg-sqi-infer verify-bundles
```

`verify-bundles` checks the packaged model files against their recorded SHA-256 hashes.

## Run inference

Mount an input/output directory at `/data`, then choose one model:

```bash
docker run --rm -v /host/data:/data ecg-sqi-infer predict --model 12lead-conformer  --input /data/input --fs 500 --out /data/out/12lead-conformer
docker run --rm -v /host/data:/data ecg-sqi-infer predict --model singlelead-conformer --input /data/input --fs 500 --out /data/out/singlelead-conformer
docker run --rm -v /host/data:/data ecg-sqi-infer predict --model 12lead-rbfsvm      --input /data/input --fs 500 --out /data/out/12lead-rbfsvm
docker run --rm -v /host/data:/data ecg-sqi-infer predict --model singlelead-rbfsvm   --input /data/input --fs 500 --out /data/out/singlelead-rbfsvm
```

`--fs` is the input sampling frequency in hertz. The service resamples to 125 Hz and predicts non-overlapping 10-second windows.

## Inputs

`--input` accepts one file or recursively scans a directory:

- `.npy`: numeric ECG array.
- `.npz`: numeric array under `signal`, `ecg`, `x`, `X`, `sig`, `sig_125`, or `signals`.
- `.csv`: numeric samples, one lead per column.
- WFDB: pass the `.hea` file; keep its companion signal files beside it.

Single-lead models accept `(samples,)`, `(samples, 1)`, or `(1, samples)`. Twelve-lead models accept `(samples, 12)` or `(12, samples)`.

Each run writes `<record_id>_segments.csv`, `all_segments.csv`, and `run_summary.json` below `--out`. Conformer and RBF-SVM result rows include the predicted class and `prob_*` columns.

## WSL example

Windows drives are available below `/mnt` in WSL. For `E:\ecg-data`:

```bash
docker run --rm -v /mnt/e/ecg-data:/data ecg-sqi-infer \
  predict --model singlelead-conformer --input /data/input --fs 500 --out /data/out
```

Quote the volume argument when its path contains spaces, for example `-v "/mnt/e/My Data:/data"`.
