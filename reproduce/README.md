# Reproduce Readiness Checks

This folder is a smoke/audit layer, not the main experiment pipeline.

It verifies that report tables, figure source data, and report-generation
entrypoints are present and runnable without writing to the main `outputs/`
tree.

## Commands

```bash
python reproduce/check_reproduce.py artifact
python reproduce/check_reproduce.py chapter4-render
python reproduce/check_reproduce.py public-smoke
python reproduce/check_reproduce.py all
```

All generated logs and summaries are written under `reproduce/work/`.

`public-smoke` uses a bounded public BUTQDB remote-range read so a fresh clone
can verify the clean-data entrypoint without downloading the full database. A
`historical_support_exact=false` result is an expected warning, not a
replacement for the frozen Chapter 4 numbers or a full public-data rebuild.
