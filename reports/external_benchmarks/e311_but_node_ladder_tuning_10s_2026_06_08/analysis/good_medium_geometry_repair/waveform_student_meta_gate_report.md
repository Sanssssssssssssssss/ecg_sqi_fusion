# Waveform Student Synthetic-Only Meta-Gate

The gate is trained only on synthetic validation probabilities from fixed waveform-only branches. BUT buckets are report-only.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_val_meta_gate | synthetic_val | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0 |
| synthetic_val_meta_gate | synthetic_test | 0.998462 | 0.998203 | 0.995816 | 1.000000 | 0.995851 | 2 | 0 | 1 |
| synthetic_val_meta_gate | original_test_all_10s+ | 0.796744 | 0.699911 | 0.888462 | 0.761862 | 0.360097 | 406 | 1000 | 42 |
| synthetic_val_meta_gate | original_all_10s+ | 0.796608 | 0.829689 | 0.701754 | 0.878152 | 0.938505 | 5082 | 1219 | 102 |
| synthetic_val_meta_gate | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| synthetic_val_meta_gate | bad_outlier_stress | 0.099315 | 0.060228 | 0.000000 | 0.000000 | 0.099315 | 0 | 0 | 42 |
