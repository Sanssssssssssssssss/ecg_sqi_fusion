# BUT 10s Morphology Refine Generator

Narrow morphology refine around s02 and mix05. The aim is not SNR; it is expert-style morphology: medium QRS usable, bad QRS confounded.

| rank | spec | cw | return | BUT acc | BUT bal | macro-F1 | recalls good/medium/bad | PTB acc | PTB bad | denoise | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | r06_s02_calibration_guard | 1.20,1.50,1.80 | 0 | 0.7018 | 0.7606 | 0.6522 | 0.694/0.690/0.898 | 0.9637 | 1.0000 | 2.624 | Explicitly rescue good while keeping s02 morphology. |
| 2 | r03_s02_good_medium_balance | 1.08,1.58,1.76 | 0 | 0.7373 | 0.8031 | 0.6699 | 0.780/0.683/0.946 | 0.9528 | 0.9986 | 2.609 | Middle point between s02 and mix05. |
| 3 | r04_mix05_plus_bad_spikes | 1.05,1.60,1.80 | 0 | 0.7133 | 0.7990 | 0.6542 | 0.945/0.500/0.951 | 0.9777 | 1.0000 | 2.858 | mix05-like medium with stronger bad QRS-confounding spikes. |
| 4 | r05_mix05_good_guard_bad_hard | 1.12,1.56,1.84 | 0 | 0.6945 | 0.6438 | 0.5941 | 0.976/0.484/0.472 | 0.9482 | 0.9986 | 2.481 | Try to retain mix05 balanced score while restoring bad. |
| 5 | r02_s02_less_bad_pressure | 1.10,1.52,1.74 | 0 | 0.6644 | 0.7463 | 0.6142 | 0.992/0.376/0.871 | 0.9655 | 1.0000 | 2.666 | Reduce over-bad calibration while keeping morphology events. |
| 6 | r01_s02_good_guard | 1.15,1.48,1.78 | 0 | 0.7387 | 0.5379 | 0.5435 | 0.785/0.763/0.066 | 0.9591 | 0.9946 | 2.619 | s02-style bad spikes with stronger good/QRS guard. |
