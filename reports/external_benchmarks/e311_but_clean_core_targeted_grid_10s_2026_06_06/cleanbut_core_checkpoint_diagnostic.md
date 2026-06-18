# CleanBUT Core Checkpoint Diagnostic

This diagnostic evaluates trained checkpoints on CleanBUT-core balanced subsets. It is not the formal BUT benchmark.

Formal CleanBUT test split caveat: clean-core test has no bad samples, so all-split balanced diagnostics are used for three-class checks.

| subset | variant | acc | macro-F1 | recalls good/medium/bad | orig BUT macro | clean64 score |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| clean_strict_all_balanced | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d8774356567f` | 0.9935 | 0.9935 | 0.999/0.982/1.000 | 0.4173 | 0.2922 |
| clean_strict_all_balanced | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d7274ab2f04d` | 0.9875 | 0.9875 | 0.999/0.972/0.992 | 0.4314 | 0.2931 |
| clean_strict_all_balanced | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_238aae411c7e` | 0.9796 | 0.9796 | 0.999/0.940/1.000 | 0.4249 | 0.2985 |
| clean_strict_all_balanced | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_6bb3561abb9a` | 0.9750 | 0.9749 | 0.926/0.999/1.000 | 0.5624 | 0.2985 |
| clean_strict_all_balanced | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_adf610ec036d` | 0.9638 | 0.9637 | 1.000/0.892/1.000 | 0.4195 | 0.2931 |
| clean_strict_all_balanced | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_9683ffc33626` | 0.9175 | 0.9162 | 1.000/0.752/1.000 | 0.4999 | 0.2922 |
| clean_target_plus_strict_all_balanced | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d7274ab2f04d` | 0.9730 | 0.9730 | 0.986/0.942/0.992 | 0.4314 | 0.2931 |
| clean_target_plus_strict_all_balanced | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d8774356567f` | 0.9683 | 0.9683 | 0.970/0.935/1.000 | 0.4173 | 0.2922 |
| clean_target_plus_strict_all_balanced | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_238aae411c7e` | 0.9597 | 0.9596 | 0.988/0.891/1.000 | 0.4249 | 0.2985 |
| clean_target_plus_strict_all_balanced | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_6bb3561abb9a` | 0.9502 | 0.9500 | 0.856/0.994/1.000 | 0.5624 | 0.2985 |
| clean_target_plus_strict_all_balanced | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_adf610ec036d` | 0.9432 | 0.9428 | 0.998/0.831/1.000 | 0.4195 | 0.2931 |
| clean_target_plus_strict_all_balanced | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_9683ffc33626` | 0.9023 | 0.9001 | 1.000/0.707/1.000 | 0.4999 | 0.2922 |
