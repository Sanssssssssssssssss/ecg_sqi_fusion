# BUT 10s Morphology V2 Generator

Small morphology-only continuation.  Formal protocol remains 10s P1; thresholds are validation-only.

| rank | spec | cw | return | BUT acc | BUT bal | macro-F1 | recalls good/medium/bad | PTB acc | PTB bad | denoise | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | v204_medium_pt_strong_bad_b10 | 1.08,1.62,1.82 | 0 | 0.7000 | 0.6714 | 0.6300 | 0.983/0.481/0.550 | 0.9700 | 0.9973 | 2.744 | Test whether stronger P/T/ST unreliability can lift medium without losing bad. |
| 2 | v203_s02_good_rescue | 1.18,1.50,1.84 | 0 | 0.6876 | 0.7618 | 0.6643 | 0.917/0.480/0.888 | 0.9709 | 0.9986 | 2.745 | S02-style bad/medium coexistence with explicit good rescue. |
| 3 | v201_b10_medium_local_bad_confuse | 1.06,1.56,1.84 | 0 | 0.6980 | 0.6459 | 0.6222 | 0.960/0.503/0.474 | 0.9600 | 0.9986 | 2.620 | B10 macro baseline plus local medium morphology and QRS-confusing bad. |
| 4 | v206_triplet_margin | 1.12,1.56,1.88 | 0 | 0.6381 | 0.7382 | 0.5646 | 0.891/0.404/0.920 | 0.9673 | 0.9986 | 2.691 | Wider morphology margin: good mild, medium local, bad QRS-confounded. |
| 5 | v202_mix05_bad_repair | 1.04,1.58,1.86 | 0 | 0.7194 | 0.6228 | 0.6078 | 0.985/0.535/0.348 | 0.9591 | 0.9973 | 2.667 | Keep mix05 medium cues but restore bad through dense pseudo-QRS events. |
| 6 | v205_good_overlap_medium_boundary | 1.22,1.48,1.86 | 0 | 0.5188 | 0.4388 | 0.3960 | 0.126/0.859/0.331 | 0.9732 | 0.9986 | 2.806 | Make good less pristine so BUT good is not rejected, while bad stays hard. |
