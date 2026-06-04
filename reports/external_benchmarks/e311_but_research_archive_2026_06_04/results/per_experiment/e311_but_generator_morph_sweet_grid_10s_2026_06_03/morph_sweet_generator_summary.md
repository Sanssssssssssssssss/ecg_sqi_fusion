# BUT 10s Morphology Sweet-Spot Generator

Morphology-focused synthetic rules.  SNR is not the design axis; local shape events and QRS interpretability are.

| rank | spec | cw | return | BUT acc | BUT bal | macro-F1 | recalls good/medium/bad | PTB acc | PTB bad | denoise | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | s02_dense_bad_spikes_medium_soft | 1,1.50,1.88 | 0 | 0.7464 | 0.7828 | 0.6868 | 0.763/0.722/0.864 | 0.9682 | 1.0000 | 2.647 | Bad has dense QRS-confounding spikes; medium stays soft. |
| 2 | s03_but_medium_motion | 1,1.62,1.80 | 0 | 0.6511 | 0.6514 | 0.5863 | 0.515/0.760/0.679 | 0.9605 | 0.9986 | 2.561 | Aggressive BUT-like medium motion while keeping QRS mostly visible. |
| 3 | s04_contact_bad_medium_visible | 1,1.56,1.88 | 0 | 0.7438 | 0.7792 | 0.6840 | 0.854/0.644/0.839 | 0.9646 | 0.9986 | 2.556 | Bad contact/flatline, medium visible-QRS morphology uncertainty. |
| 4 | s06_medium_local_morph_bad_hard | 1,1.60,1.86 | 0 | 0.6294 | 0.5644 | 0.5523 | 0.945/0.395/0.353 | 0.9609 | 0.9959 | 2.541 | Hard bad plus explicit local morphology medium. |
| 5 | s05_medium_ambiguous_good_guard | 1,1.58,1.82 | 0 | 0.5411 | 0.4779 | 0.4798 | 0.509/0.586/0.338 | 0.9809 | 0.9946 | 2.857 | Good less pristine, medium ambiguous, bad QRS-confounded. |
