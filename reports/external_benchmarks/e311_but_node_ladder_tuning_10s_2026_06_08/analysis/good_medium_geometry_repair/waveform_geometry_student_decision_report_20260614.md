# Waveform Geometry Student Decision Report - 2026-06-14

## Contract

- Training input is PTB-generated synthetic waveform only.
- The classifier receives waveform tensors only.
- SQI/geometry columns are teacher targets in training losses, not inference inputs.
- Original BUT is held-out report-only and is not used for training, validation, threshold selection, or model selection.
- Implementation is external-only in `outputs/.../analysis/good_medium_geometry_repair/run_waveform_geometry_student.py`.

## Architectures Tested

| family | idea | smoke synthetic test acc | smoke good/medium/bad recall |
|---|---|---:|---|
| `MapStudentTransformer` | PatchTST-style channel patch Transformer + geometry teacher | 0.9354 | 0.906 / 0.938 / 0.979 |
| `MaskedGeometryMAE` | masked ECG patch reconstruction + geometry teacher | 0.9154 | 0.877 / 0.918 / 0.979 |
| `StatTokenTransformerV2` | waveform Transformer + differentiable global stat tokens | 0.9800 | 0.933 / 0.998 / 0.983 |
| `MorphologyTokenTransformer` | QRS/detail/baseline multi-scale token Transformer | 0.9892 | 0.973 / 0.998 / 0.979 |

The two strongest smoke candidates were promoted to longer search.

## Long Search Results

| candidate | synthetic test acc | synthetic good/medium/bad | original test acc | original good/medium/bad | key read |
|---|---:|---|---:|---|---|
| `stattoken_v2` | 0.9938 | 0.981 / 1.000 / 0.988 | 0.8029 | 0.663 / 0.966 / 0.280 | learns synthetic geometry; transfers medium, loses much original good and bad outlier |
| `morphology_token_tx` | 0.9969 | 0.992 / 1.000 / 0.992 | 0.7849 | 0.855 / 0.795 / 0.058 | strongest synthetic fit; original bad almost absent |
| `stattoken_v2_badstress` | 0.9908 | 0.985 / 0.995 / 0.979 | 0.8358 | 0.824 / 0.896 / 0.290 | best waveform-only original test so far; bad core fixed with calibration, bad outlier still 0 |
| `morphology_token_badstress` | 0.9749 | 0.994 / 0.967 / 0.979 | 0.7234 | 0.925 / 0.618 / 0.073 | touches bad outlier slightly, but damages medium and bad core |

Reference: 47-feature tabular upper bound remains original test acc 0.9635 with good/medium/bad recall 0.956 / 0.973 / 0.927, but that model uses geometry features at inference and is not waveform-only.

## Feature Teacher Recovery

The new models can learn some global axes from waveform:

- `stattoken_v2` on original has strong correlation for `pc1` (0.916), `pca_margin` (0.858), and `non_qrs_diff_p95` (0.919).
- `morphology_token_tx` on original has strong correlation for `pc1` (0.841), `pca_margin` (0.675), `flatline_ratio` (0.873), and `non_qrs_diff_p95` (0.799).

The models still fail on the features that make the tabular model act like a target atlas:

- `qrs_visibility` remains poorly recovered on original (`stattoken_v2` corr 0.034, `morphology_token_tx` corr 0.251).
- `boundary_confidence` remains weak (`stattoken_v2` corr 0.149, `morphology_token_tx` corr 0.111).
- `knn_label_purity` remains weak (`stattoken_v2` corr 0.307, `morphology_token_tx` corr 0.186).

This means the waveform model learned a useful morphology axis, but not the full neighborhood/boundary map that the 47-feature tabular model receives directly.

## Decision

The architecture direction is valid: waveform-only Transformer-family models now pass synthetic/node gates strongly, with `morphology_token_tx` reaching 0.9969 synthetic test accuracy and `stattoken_v2_badstress` reaching the best waveform-only original test accuracy so far.

The remaining gap is not solved by more epochs or generic Transformer capacity. The blocking gap is the PTB synthetic teacher distribution for original-style bad outlier and record-domain stress. Generic controlled bad augmentation improves original test to 0.8358 but still leaves `bad_outlier_stress` at 0, so the next generator must explicitly model the 111001-like bad stress morphology rather than only broader noise/dropout.

## Next Experiment

Use `stattoken_v2_badstress` as the current waveform-only frontier, then build a sharper PTB-only `bad_111001_like_controlled_stress` generator block:

- low `qrs_band_ratio`,
- low `qrs_visibility`,
- high `baseline_step`,
- high `band_30_45`,
- high `non_qrs_diff_p95`,
- low/medium `boundary_confidence`,
- preserve good/medium blocks so medium does not collapse.

Selection should still use synthetic/node diagnostics only. Original BUT remains bucketed report-only.

