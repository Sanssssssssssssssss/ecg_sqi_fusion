# wfdb-qrs-kit

[`wfdb-qrs-kit`](https://github.com/Sanssssssssssssssss/wfdb-qrs-kit) is the
companion Python library written for this project. It provides robust wrappers
and setup utilities for WFDB-compatible QRS detectors, including the paper-era
`wqrs` and EP Limited/Hamilton detectors used by the classical SQI workflow.

```bash
pip install wfdb-qrs-kit
wfdb-qrs-kit doctor
```

To use locally installed detector executables:

```bash
wfdb-qrs-kit install-detectors --from-bin-dir /path/to/wfdb/bin --no-download
```

The package handles executable discovery, ECG/WFDB conversion, per-lead
detection, and structured results. Its Python wrapper is MIT licensed. External
detector programs keep their original upstream GPL/LGPL terms and are not
bundled in the default wheel.

Full API and detector setup documentation is available on the
[`wfdb-qrs-kit` site](https://Sanssssssssssssssss.github.io/wfdb-qrs-kit/).

