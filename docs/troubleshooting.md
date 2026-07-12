# Troubleshooting

## A bundle fails verification

Run:

```bash
python -m src.ecg_sqi_inference verify-bundles
```

Do not load a model whose recorded hash fails. Restore the file from the same
repository revision rather than editing the manifest.

## The input has the wrong lead shape

Twelve-lead models require exactly 12 leads; single-lead models require one.
Transpose `(12, samples)` inputs or select the intended lead before inference.
The library performs orientation only when one dimension unambiguously equals
the required lead count.

## A record produces no segments

After resampling to 125 Hz, a record must contain at least 1,250 samples. The
pipeline intentionally does not pad short signals.

## Paper detectors are not found

```bash
wfdb-qrs-kit doctor
wfdb-qrs-kit install-detectors --from-bin-dir /path/to/wfdb/bin --no-download
```

Executable discovery and WSL compilation are covered in the
[supporting-library guide](wfdb_qrs_kit.md).

## A public rebuild differs from the frozen report

Check the generated audit summary. `historical_support_exact=false` means the
public support pool differs from the archived v116 pool. Treat the run as a
pipeline validation, not an exact numerical replay.

## Docker cannot see a Windows path

Use an absolute Docker Desktop path or its WSL form. Quote mounts containing
spaces, for example `-v "/mnt/e/My Data:/data"`.
