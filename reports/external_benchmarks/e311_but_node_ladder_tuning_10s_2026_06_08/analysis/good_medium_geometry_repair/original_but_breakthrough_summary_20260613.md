# Original BUT Boundary Breakthrough Summary

This is a report-only research note. Clean/SemiClean/node diagnostic remains the selection source; original BUT is used only for bucketed diagnosis.

## Clean Frontier

- `N7200_gm_trim_bad` is promoted on Clean/SemiClean/node diagnostic.
- Best clean mode: `simple_pc1_gm_gate_t226`.
- Clean/node metrics: acc `0.995185`, macro-F1 `0.994917`, good/medium/bad recall `0.999167/0.995972/0.986778`.

## Original Report-Only Progression

| mode | original test acc | macro-F1 | good | medium | bad | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `wavegood + precision_veto` | `0.897487` | `0.802400` | `0.906868` | `0.928830` | `0.476886` | First useful waveform block; good-domain rescue. |
| `largeblock_diagnostic` | `0.903740` | `0.821791` | `0.903571` | `0.936512` | `0.552311` | Adds broad medium guard and bad-stress precision rule. |
| `axis_diagnostic` | `0.908104` | `0.839910` | `0.908242` | `0.933800` | `0.630170` | Adds ECG-style spectral/autocorrelation axes; current best balanced diagnostic. |
| `axis2_diagnostic` | `0.910228` | `0.847387` | `0.908242` | `0.934026` | `0.671533` | Best test score, but original-all acc drops; likely more overfit. |

Bucket notes for `axis_diagnostic`:

- `original_test_good_medium_only`: acc `0.922266`.
- `original_test_drop_bad_outlier_reference`: acc `0.923396`.
- `original_test_bad_core_near_boundary`: bad recall `1.000000`.
- `original_test_bad_outlier_stress`: bad recall `0.479452`.
- `original_all_bad_outlier_stress`: bad recall `0.829309`.

## What Actually Helped

- The biggest useful good rescue is not SNR-only; it needs waveform morphology: low `pc1`, high `pc2`, low non-QRS derivative, high low-frequency ratio, and sparse high-amplitude spikes.
- Bad stress improves with simple ECG quality axes: low `template_corr`, low autocorrelation peak, high low-frequency power, and flatline/baseline shape.
- Good/medium remaining errors are now mostly local `111001` overlap:
  - `111001 good->medium`: `246`.
  - `111001 medium->good`: `213`.
  - `111001 bad->medium`: `140`.
  - `125001 good->medium`: `75`.

## Important Caution

`axis2_diagnostic` improves original-test acc but hurts original-all medium recall. Treat it as a stress-search artifact, not the main conclusion. The safer research conclusion is `axis_diagnostic`: it improves bad stress and good/medium balance while preserving broader original-all behavior better.

## Current Working Hypothesis

The remaining gap is not Clean/SemiClean model capacity. N7200 is already essentially solved on the node diagnostic. The original gap is a domain/label-boundary problem concentrated in:

- `111001` good/medium outlier-low-confidence overlap.
- `111001` bad outlier stress, which does not look like the clean bad core.
- `125001` good-domain shift with weak QRS/detail but label still good.

Next useful work should prefer one or two stable morphology/SQI axes or a domain-adaptation block, not many tiny threshold patches.
