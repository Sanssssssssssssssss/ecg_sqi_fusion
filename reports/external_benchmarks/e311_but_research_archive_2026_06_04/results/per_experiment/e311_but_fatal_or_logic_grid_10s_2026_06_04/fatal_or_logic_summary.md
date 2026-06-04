# BUT 10s Fatal-OR Logic Grid

Hypothesis: BUT bad behaves like OR(any fatal usability dimension fails), good behaves like AND(all critical dimensions are usable), and medium is an independent QRS-usable/detail-unreliable cluster.

Anchor b10: acc 0.7735, balanced 0.8045, macro-F1 0.7238, recalls 0.824/0.724/0.866.

| rank | mode | spec | family | cw | BUT acc | bal | macro-F1 | recalls good/medium/bad | min med/bad | PTB acc | PTB bad | denoise | feature score | note |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | quick | medium_independent_03 | medium_independent | 1.12,1.66,1.86 | 0.7650 | 0.7336 | 0.7377 | 0.947/0.628/0.625 | 0.625 | 0.9714 | 0.9973 | 2.747 | 0.765 | Medium as independent detail-unreliable cluster with QRS preserved. |
| 2 | full | fatal_or_qrs_confuse_06 | fatal_or_qrs_confuse | 1.10,1.56,1.94 | 0.8064 | 0.7471 | 0.7319 | 0.842/0.796/0.603 | 0.603 | 0.9827 | 1.0000 | 2.945 | 0.766 | Bad as fatal QRS-confusion OR, with QRS-visible detail-unreliable medium. |
| 3 | full | medium_independent_04 | medium_independent | 1.10,1.64,1.90 | 0.7591 | 0.7574 | 0.7277 | 0.880/0.663/0.730 | 0.663 | 0.9841 | 1.0000 | 3.038 | 0.765 | Medium as independent detail-unreliable cluster with QRS preserved. |
| 4 | quick | medium_independent_04 | medium_independent | 1.10,1.64,1.90 | 0.7391 | 0.7350 | 0.7198 | 0.978/0.548/0.679 | 0.548 | 0.9664 | 0.9986 | 2.648 | 0.765 | Medium as independent detail-unreliable cluster with QRS preserved. |
| 5 | full | medium_independent_03 | medium_independent | 1.12,1.66,1.86 | 0.7686 | 0.6727 | 0.6887 | 0.901/0.692/0.426 | 0.426 | 0.9827 | 0.9973 | 3.031 | 0.765 | Medium as independent detail-unreliable cluster with QRS preserved. |
| 6 | quick | fatal_or_qrs_confuse_06 | fatal_or_qrs_confuse | 1.10,1.56,1.94 | 0.7161 | 0.6691 | 0.6718 | 0.634/0.796/0.577 | 0.577 | 0.9768 | 0.9986 | 2.668 | 0.766 | Bad as fatal QRS-confusion OR, with QRS-visible detail-unreliable medium. |
| 7 | quick | fatal_or_qrs_confuse_02 | fatal_or_qrs_confuse | 1.08,1.52,1.96 | 0.7322 | 0.7790 | 0.6643 | 0.987/0.513/0.837 | 0.513 | 0.9673 | 1.0000 | 2.657 | 0.767 | Bad as fatal QRS-confusion OR, with QRS-visible detail-unreliable medium. |
| 8 | quick | fatal_or_qrs_confuse_04 | fatal_or_qrs_confuse | 1.10,1.58,1.90 | 0.6765 | 0.7005 | 0.6528 | 0.946/0.453/0.703 | 0.453 | 0.9723 | 0.9986 | 2.664 | 0.766 | Bad as fatal QRS-confusion OR, with QRS-visible detail-unreliable medium. |
| 9 | quick | good_and_medium_shell_02 | good_and_medium_shell | 1.04,1.64,1.92 | 0.7054 | 0.6851 | 0.6523 | 0.946/0.519/0.591 | 0.519 | 0.9705 | 0.9986 | 2.626 | 0.767 | Good as all-critical-dimensions-good; medium absorbs imperfect usable cases. |
| 10 | quick | fatal_or_contact_01 | fatal_or_contact | 1.10,1.56,1.92 | 0.7841 | 0.6631 | 0.6432 | 0.944/0.693/0.353 | 0.353 | 0.9723 | 0.9973 | 2.636 | 0.766 | Bad as contact/flat fatal OR, not all-bad-at-once. |
| 11 | quick | fatal_or_contact_02 | fatal_or_contact | 1.10,1.58,1.92 | 0.6796 | 0.6764 | 0.6402 | 0.965/0.451/0.613 | 0.451 | 0.9768 | 1.0000 | 2.686 | 0.767 | Bad as contact/flat fatal OR, not all-bad-at-once. |
| 12 | quick | fatal_or_qrs_confuse_05 | fatal_or_qrs_confuse | 1.06,1.50,1.98 | 0.6834 | 0.7442 | 0.6397 | 0.902/0.489/0.842 | 0.489 | 0.9664 | 1.0000 | 2.698 | 0.764 | Bad as fatal QRS-confusion OR, with QRS-visible detail-unreliable medium. |
| 13 | full | fatal_or_qrs_confuse_02 | fatal_or_qrs_confuse | 1.08,1.52,1.96 | 0.6758 | 0.6800 | 0.6225 | 0.980/0.430/0.630 | 0.430 | 0.9846 | 1.0000 | 2.991 | 0.767 | Bad as fatal QRS-confusion OR, with QRS-visible detail-unreliable medium. |
| 14 | quick | good_and_medium_shell_03 | good_and_medium_shell | 1.06,1.66,1.90 | 0.6912 | 0.6589 | 0.6157 | 0.992/0.459/0.526 | 0.459 | 0.9750 | 0.9986 | 2.680 | 0.766 | Good as all-critical-dimensions-good; medium absorbs imperfect usable cases. |
| 15 | quick | fatal_or_qrs_confuse_03 | fatal_or_qrs_confuse | 1.10,1.56,1.92 | 0.7155 | 0.6374 | 0.6134 | 0.963/0.540/0.409 | 0.409 | 0.9700 | 0.9986 | 2.663 | 0.765 | Bad as fatal QRS-confusion OR, with QRS-visible detail-unreliable medium. |
| 16 | quick | medium_independent_01 | medium_independent | 1.08,1.62,1.88 | 0.6642 | 0.6163 | 0.6129 | 0.726/0.629/0.494 | 0.494 | 0.9696 | 0.9973 | 2.638 | 0.765 | Medium as independent detail-unreliable cluster with QRS preserved. |
| 17 | quick | fatal_or_contact_04 | fatal_or_contact | 1.08,1.58,1.94 | 0.6150 | 0.6027 | 0.5864 | 0.877/0.408/0.523 | 0.408 | 0.9728 | 0.9986 | 2.598 | 0.765 | Bad as contact/flat fatal OR, not all-bad-at-once. |
| 18 | quick | good_and_medium_shell_04 | good_and_medium_shell | 1.08,1.64,1.94 | 0.6595 | 0.6536 | 0.5781 | 0.833/0.522/0.606 | 0.522 | 0.9728 | 0.9986 | 2.676 | 0.766 | Good as all-critical-dimensions-good; medium absorbs imperfect usable cases. |
| 19 | quick | medium_independent_02 | medium_independent | 1.10,1.64,1.88 | 0.7085 | 0.5777 | 0.5717 | 0.987/0.525/0.221 | 0.221 | 0.9759 | 0.9973 | 2.744 | 0.765 | Medium as independent detail-unreliable cluster with QRS preserved. |
| 20 | quick | good_and_medium_shell_01 | good_and_medium_shell | 1.02,1.62,1.92 | 0.6923 | 0.6862 | 0.6162 | 0.956/0.482/0.620 | 0.482 | 0.9596 | 0.9986 | 2.616 | 0.767 | Good as all-critical-dimensions-good; medium absorbs imperfect usable cases. |
| 21 | quick | fatal_or_contact_03 | fatal_or_contact | 1.12,1.56,1.90 | 0.6741 | 0.6403 | 0.6038 | 0.998/0.424/0.499 | 0.424 | 0.9628 | 1.0000 | 2.561 | 0.768 | Bad as contact/flat fatal OR, not all-bad-at-once. |
| 22 | quick | fatal_or_qrs_confuse_01 | fatal_or_qrs_confuse | 1.08,1.54,1.92 | 0.7100 | 0.5765 | 0.5796 | 0.876/0.618/0.236 | 0.236 | 0.9596 | 0.9986 | 2.577 | 0.764 | Bad as fatal QRS-confusion OR, with QRS-visible detail-unreliable medium. |
