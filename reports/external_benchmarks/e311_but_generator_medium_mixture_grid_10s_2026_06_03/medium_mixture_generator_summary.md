# BUT 10s Medium Mixture Generator

Mixture-style medium generation after uniform medium/bad rules failed to beat b10. Formal BUT protocol remains 10s P1; calibration is validation-only.

| rank | spec | cw | return | BUT acc | BUT bal | macro-F1 | recalls good/medium/bad | PTB acc | PTB bad | denoise | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | mix05_medium_protocol_boundary | 1,1.60,1.78 | 0 | 0.7747 | 0.8111 | 0.7018 | 0.860/0.695/0.878 | 0.9678 | 0.9986 | 2.629 | Mixed expert-boundary medium: several mild failure modes. |
| 2 | mix02_medium_contact_short | 1,1.52,1.84 | 0 | 0.7518 | 0.7747 | 0.7122 | 0.864/0.654/0.805 | 0.9696 | 1.0000 | 2.634 | Medium can have short contact loss, but QRS is preserved. |
| 3 | mix06_medium_strong_bad_soft | 1,1.62,1.70 | 0 | 0.6491 | 0.6129 | 0.5669 | 0.508/0.774/0.557 | 0.9578 | 0.9986 | 2.598 | Soften bad and strengthen medium boundary cues. |
| 4 | mix03_medium_lowamp_visible_qrs | 1,1.54,1.82 | 0 | 0.7656 | 0.6945 | 0.6818 | 0.939/0.648/0.496 | 0.9673 | 0.9986 | 2.621 | BUT-like low amplitude medium with visible QRS. |
| 5 | mix08_b10_medium_bad_prior_guard | 1,1.56,1.88 | 0 | 0.6700 | 0.7345 | 0.6323 | 0.955/0.421/0.827 | 0.9582 | 0.9986 | 2.517 | Preserve high bad pressure while adding heterogeneous medium. |
| 6 | mix01_b10_medium_hetero_light | 1,1.48,1.86 | 0 | 0.6633 | 0.6730 | 0.6228 | 0.972/0.412/0.635 | 0.9709 | 0.9986 | 2.602 | Near b10 bad with heterogeneous but light medium. |
| 7 | mix07_medium_good_overlap | 1,1.50,1.82 | 0 | 0.6692 | 0.7118 | 0.6518 | 0.991/0.397/0.747 | 0.9732 | 0.9986 | 2.783 | Make good less pristine without collapsing medium to bad. |
| 8 | mix04_medium_pt_unreliable | 1,1.58,1.80 | 0 | 0.7938 | 0.6543 | 0.6702 | 0.913/0.741/0.309 | 0.9641 | 0.9973 | 2.611 | Class-2 emphasis: P/T/ST unreliable while QRS stays usable. |
