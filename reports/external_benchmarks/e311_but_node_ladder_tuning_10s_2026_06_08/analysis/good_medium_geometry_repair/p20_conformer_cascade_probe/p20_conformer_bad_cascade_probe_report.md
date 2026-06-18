# P20 + Conformer Bad-Stress Cascade Probe

Report-only oracle sweep on original thresholds. Original is not used for training or selection.

## best_acc threshold=0.9990
- original_test_all_10s+: acc=0.8233, mf1=0.7028, g/m/b=0.858/0.843/0.302, g->bad=5, m->bad=27, b->g/m=190/97
- original_all_10s+: acc=0.8284, mf1=0.8549, g/m/b=0.745/0.910/0.935, g->bad=10, m->bad=36, b->g/m=191/155
- bad_core_nearboundary: acc=1.0000, mf1=0.3333, g/m/b=0.000/0.000/1.000, g->bad=0, m->bad=0, b->g/m=0/0
- bad_outlier_stress: acc=0.0171, mf1=0.0112, g/m/b=0.000/0.000/0.017, g->bad=0, m->bad=0, b->g/m=190/97

## best_balanced threshold=0.9990
- original_test_all_10s+: acc=0.8233, mf1=0.7028, g/m/b=0.858/0.843/0.302, g->bad=5, m->bad=27, b->g/m=190/97
- original_all_10s+: acc=0.8284, mf1=0.8549, g/m/b=0.745/0.910/0.935, g->bad=10, m->bad=36, b->g/m=191/155
- bad_core_nearboundary: acc=1.0000, mf1=0.3333, g/m/b=0.000/0.000/1.000, g->bad=0, m->bad=0, b->g/m=0/0
- bad_outlier_stress: acc=0.0171, mf1=0.0112, g/m/b=0.000/0.000/0.017, g->bad=0, m->bad=0, b->g/m=190/97
