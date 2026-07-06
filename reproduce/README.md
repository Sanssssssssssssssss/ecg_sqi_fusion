# Full Reproduction Targets

`reproduce/` is an audit reproduction layer. It runs named report targets and
does not write the frozen paper evidence directory.

Each target starts from the repository code, writes under `reproduce/work/`, and
emits:

- `summary.json`
- `summary.md`
- `audit_matrix.csv`
- `artifact_checksums.csv`

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

## Supplemental Targets

```powershell
powershell -ExecutionPolicy Bypass -File reproduce/scripts/ps1/supplemental/sqi_supplemental.ps1
powershell -ExecutionPolicy Bypass -File reproduce/scripts/ps1/supplemental/transformer_supplemental.ps1
```

```bash
bash reproduce/scripts/sh/supplemental/sqi_supplemental.sh
bash reproduce/scripts/sh/supplemental/transformer_supplemental.sh
```

The supplemental targets cover report-side analyses such as fSQI scans, strict
Table 6, domain/generalization checks, and transformer M6-M9 counterfactual
figures.

## Notes

- Output root: `reproduce/work/<target>/<run_id>/`.
- Downloaded public data root: `reproduce/work/data/`.
- `ECG_E31_PREDICTION_ROOT` defaults to the tracked frozen v116 prediction
  artifact for CPU-lane Conformer audits.
- A clean public rebuild is expected to be cleanly runnable, but it is not
  claimed to be an exact byte replay of the frozen v116 support pool.
