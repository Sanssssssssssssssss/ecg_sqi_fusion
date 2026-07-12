# Data Availability

Raw ECG databases and generated experiment arrays are not redistributed.

| Dataset | Provider | Pipeline role | Local root |
|---|---|---|---|
| PhysioNet/CinC Challenge 2011 Set-A | PhysioNet | Classical reproduction and 12-lead waveform evaluation | `data/physionet/challenge-2011/set-a` |
| MIT-BIH Noise Stress Test Database | PhysioNet | EM/MA controls | `data/physionet/nstdb` |
| PTB-XL | PhysioNet | Clean carriers for train-only proposals | managed by transformer data setup |
| BUT QDB | PhysioNet | Native graded single-lead evaluation | managed by transformer data setup |

The pipelines download public data when permitted or accept an existing local
copy. `data/`, `outputs/`, and `reproduce/work/` are ignored by Git.

## Submitted binary assets

Only four compact inference artifacts and their profiles are versioned. Their
hashes are frozen in `pretrained/inference/manifest.json` and documented in the
[model catalogue](models.md). They are included so the Docker inference image
can run without a separate model service.

## Licensing boundary

The repository code is MIT licensed. Public datasets retain their provider
terms. The Python `wfdb-qrs-kit` wrapper is MIT licensed, while copied or
compiled `wqrs` and EP Limited/Hamilton programs retain their upstream
GPL/LGPL-family licences; they are not bundled in the default Python wheel.

## Frozen-support caveat

The exact historical v116 support pool is not distributed. Public rebuilds are
valid smoke checks unless the audit reports
`historical_support_exact=true`; otherwise they must not be described as exact
replay of the frozen evidence.

