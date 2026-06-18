# Waveform Bad Router Probe

Report-only original-test diagnostic. Base prediction is `featurefirst_top20_hardrec_a050`; record111 candidate bad score may override to bad. This is not model selection.

| acc | macro_f1 | good_recall | medium_recall | bad_recall | bad_outlier_recall | bad_core_recall | nonbad_false_bad | threshold | mode | candidate | test_gate_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.844992 | 0.714039 | 0.848352 | 0.894261 | 0.284672 | 0.006849 | 0.966387 | 19 | 0.990000 | cand_bad | featurefirst_top20_hardrec_record111lite_a050 | 97 |
| 0.844992 | 0.714039 | 0.848352 | 0.894261 | 0.284672 | 0.006849 | 0.966387 | 19 | 0.990000 | both_bad_base002 | featurefirst_top20_hardrec_record111lite_a050 | 97 |
| 0.844992 | 0.714039 | 0.848352 | 0.894261 | 0.284672 | 0.006849 | 0.966387 | 19 | 0.990000 | cand_bad_base_uncertain096 | featurefirst_top20_hardrec_record111lite_a050 | 97 |
| 0.844520 | 0.710050 | 0.848352 | 0.894261 | 0.274939 | 0.006849 | 0.932773 | 19 | 0.970000 | both_bad_base002 | featurefirst_top20_hardrec_record111specific_a050 | 19 |
| 0.844520 | 0.710050 | 0.848352 | 0.894261 | 0.274939 | 0.006849 | 0.932773 | 19 | 0.970000 | cand_bad | featurefirst_top20_hardrec_record111specific_a050 | 19 |
| 0.844520 | 0.710050 | 0.848352 | 0.894261 | 0.274939 | 0.006849 | 0.932773 | 19 | 0.970000 | cand_bad_base_uncertain096 | featurefirst_top20_hardrec_record111specific_a050 | 19 |
| 0.844167 | 0.714205 | 0.848352 | 0.891776 | 0.294404 | 0.010274 | 0.991597 | 32 | 0.950000 | both_bad_base002 | featurefirst_top20_hardrec_record111stress_a050 | 95 |
| 0.843577 | 0.707273 | 0.848352 | 0.892454 | 0.274939 | 0.006849 | 0.932773 | 29 | 0.990000 | cand_bad | featurefirst_top20_hardrec_record111stress_a050 | 29 |
| 0.843577 | 0.712530 | 0.848352 | 0.890646 | 0.294404 | 0.010274 | 0.991597 | 38 | 0.950000 | cand_bad_base_uncertain096 | featurefirst_top20_hardrec_record111stress_a050 | 101 |
| 0.842397 | 0.710159 | 0.848352 | 0.887935 | 0.299270 | 0.017123 | 0.991597 | 54 | 0.680000 | cand_margin | featurefirst_top20_hardrec_record111stress_a050 | 146 |
| 0.840864 | 0.704525 | 0.847802 | 0.885901 | 0.294404 | 0.030822 | 0.941176 | 69 | 0.800000 | cand_margin | featurefirst_top20_hardrec_record111specific_a050 | 141 |
| 0.836145 | 0.703806 | 0.846978 | 0.873927 | 0.333333 | 0.061644 | 1.000000 | 127 | 0.800000 | cand_margin | featurefirst_top20_hardrec_record111lite_a050 | 264 |