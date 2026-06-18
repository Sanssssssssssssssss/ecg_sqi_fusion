# BUT 10s Morphology V3 Generator

Narrow follow-up after v2.  It keeps 10s P1 and validation-only calibration.

| rank | spec | cw | return | BUT acc | BUT bal | macro-F1 | recalls good/medium/bad | PTB acc | PTB bad | denoise | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | v303_mix05_medium_with_bad_floor | 1.04,1.60,1.86 | 0 | 0.5804 | 0.5771 | 0.5897 | 0.593/0.571/0.567 | 0.9673 | 0.9986 | 2.561 | Mix05-style medium, but without good-overlap and with bad floor restored. |
| 2 | v301_b10_micro_morph | 1.00,1.54,1.82 | 0 | 0.7073 | 0.7593 | 0.6710 | 0.886/0.548/0.844 | 0.9600 | 1.0000 | 2.646 | Conservative b10 continuation: tiny local morphology, do not disturb good. |
| 3 | v305_medium_guard_bad_soft | 1.08,1.66,1.78 | 0 | 0.6940 | 0.7841 | 0.6309 | 0.927/0.479/0.946 | 0.9619 | 1.0000 | 2.670 | Medium guard: lift medium with softer bad pressure. |
| 4 | v304_no_flatline_pseudo_qrs_bad | 1.06,1.54,1.88 | 0 | 0.5698 | 0.6359 | 0.5856 | 0.657/0.479/0.771 | 0.9587 | 1.0000 | 2.723 | BUT bad often has pseudo-QRS confusion rather than pure flatline. |
| 5 | v302_s02_bad_with_medium_floor | 1.08,1.58,1.84 | 0 | 0.5738 | 0.5240 | 0.5255 | 0.724/0.468/0.380 | 0.9700 | 0.9986 | 2.735 | S02 bad cue with a medium floor and less good pressure. |
