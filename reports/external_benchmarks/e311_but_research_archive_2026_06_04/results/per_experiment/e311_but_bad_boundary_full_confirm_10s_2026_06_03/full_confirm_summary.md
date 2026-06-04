# BUT 10s Bad-Boundary Full Confirmation

Full recipe confirmation for the best quick-grid bad-boundary anchors.  Formal external protocol remains 10s P1; threshold calibration remains validation-only.

| rank | id | class weight | return | BUT acc | BUT bal | macro-F1 | recalls good/medium/bad | PTB acc | PTB bad | denoise |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | b10_all_bad_wearable_full_cw190 | 1,1.40,1.90 | 0 | 0.7743 | 0.7213 | 0.6765 | 0.863/0.720/0.582 | 0.9818 | 0.9986 | 2.869 |
| 2 | r08_bad_prior_mild_full_cw160 | 1,1.42,1.60 | 0 | 0.7782 | 0.5663 | 0.5587 | 0.938/0.715/0.046 | 0.9805 | 0.9973 | 2.924 |

## Interpretation Rule

- If full b10 beats the quick b10 anchor or keeps bad recall high with better medium, continue b10-style generator adaptation.
- If full b10 does not improve over quick b10, freeze the generator rule and switch to boundary calibration / BUT-supervised head-only adaptation.
