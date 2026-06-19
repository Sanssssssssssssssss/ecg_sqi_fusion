# Clean-Val Bias Calibration Diagnostic

Report-only diagnostic for v3 waveform Transformer. Bias is selected on clean BUT val, then applied to test; this is not the PTB-only claim.

| calibration | split | bias_good | bias_medium | bias_bad | acc | good | medium | bad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_min | but_val | -0.400000 | 0.450000 | 0.000000 | 0.926313 | 0.909335 | 0.908654 | 1.000000 |
| best_min | but_test | -0.400000 | 0.450000 | 0.000000 | 0.927943 | 0.910428 | 0.912351 | 1.000000 |
| best_acc | but_val | 0.400000 | -0.100000 | 0.000000 | 0.930962 | 0.942832 | 0.864583 | 1.000000 |
| best_acc | but_test | 0.400000 | -0.100000 | 0.000000 | 0.936979 | 0.950089 | 0.872510 | 1.000000 |
| no_bias | but_val | 0.000000 | 0.000000 | 0.000000 | 0.930265 | 0.932559 | 0.880609 | 1.000000 |
| no_bias | but_test | 0.000000 | 0.000000 | 0.000000 | 0.934893 | 0.935829 | 0.890837 | 1.000000 |