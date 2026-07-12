# wfdb-qrs-kit

[`wfdb-qrs-kit`](https://github.com/Sanssssssssssssssss/wfdb-qrs-kit) is the
companion library written for this project. It provides Python wrappers and
setup utilities for WFDB-compatible QRS detectors, including the paper-era
`wqrs` and EP Limited/Hamilton programs required by the paper-aligned SQI path.

## Install and inspect

```bash
pip install wfdb-qrs-kit
wfdb-qrs-kit doctor
```

`doctor` reports which detector executables are discoverable. The search order
is explicit API argument, current environment variables, the package cache,
then system `PATH`.

## Use existing binaries

```bash
wfdb-qrs-kit install-detectors \
  --from-bin-dir /path/to/wfdb/bin \
  --no-download
```

The classical pipeline calls the wrapper for per-lead detection and receives
sample indices plus provenance metadata. This avoids embedding machine-specific
detector paths in experiment code.

## Licensing boundary

The Python wrapper, tests, examples, and documentation are MIT licensed. The
external detector programs retain their upstream licences. The default wheel
does not vendor their source or binaries; any local copy remains governed by
the upstream WFDB/EP Limited terms.

The companion [documentation site](https://Sanssssssssssssssss.github.io/wfdb-qrs-kit/)
contains its complete API, detector setup, WSL build path, and licence notes.

