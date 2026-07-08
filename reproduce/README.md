# ECG SQI Full Reproduction

This folder is the fresh-clone reproduction entrypoint. It runs this checkout
by default and writes generated data outside the repo under:

```text
../reproduce/work/<target>/<run_id>/
```

The repo `outputs/` tree is not used for reproduction outputs. BUT v116 stages
are redirected through `ECG_V116_ARTIFACTS_DIR` by the runner.

Paper SQI targets require the external `wqrs` and EP Limited/Hamilton detector
executables. If they are not already on `PATH` or in the `wfdb-qrs-kit` cache,
set:

```powershell
$env:WFDB_QRS_KIT_FROM_BIN_DIR = "E:\path\to\wfdb-qrs-bin"
```

```bash
export WFDB_QRS_KIT_FROM_BIN_DIR=/path/to/wfdb-qrs-bin
```

## Main Targets

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File reproduce/scripts/ps1/baseline_cinc2011.ps1
powershell -ExecutionPolicy Bypass -File reproduce/scripts/ps1/baseline_but.ps1
powershell -ExecutionPolicy Bypass -File reproduce/scripts/ps1/conformer_cinc2011.ps1
powershell -ExecutionPolicy Bypass -File reproduce/scripts/ps1/conformer_but.ps1
```

Linux, macOS, or WSL:

```bash
bash reproduce/scripts/sh/baseline_cinc2011.sh
bash reproduce/scripts/sh/baseline_but.sh
bash reproduce/scripts/sh/conformer_cinc2011.sh
bash reproduce/scripts/sh/conformer_but.sh
```

Direct Python:

```bash
python reproduce/run_reproduce.py --target baseline-cinc2011
python reproduce/run_reproduce.py --target baseline-but
python reproduce/run_reproduce.py --target conformer-cinc2011
python reproduce/run_reproduce.py --target conformer-but
```

Use `--raw-copy --raw-data-root <path-to-raw-data>` to junction/copy raw data
into the ignored `data/` directory. Use `--no-download` to fail if raw ECG data
download is attempted.

Each run writes `summary.json`, `summary.md`, `audit_matrix.csv`,
`artifact_checksums.csv`, and command logs under its run directory.
