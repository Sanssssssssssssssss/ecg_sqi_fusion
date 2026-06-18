# Simple Feature Medium Guard Sweep

This sweep starts from the best wide-good `pc1/qrs_prom` rule and adds one transparent medium guard: if a row is predicted good but has low QRS visibility and low PC2, return it to medium. Clean/SemiClean node gates are enforced; original BUT is report-only.

## Best Original-Test Accuracy Among Node-Viable Guards
| rule | node_acc | node_good | node_medium | acc | macro_f1 | good_recall | medium_recall | bad_recall | medium_guard_qv_max | medium_guard_pc2_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| widegood_mediumguard_000 | 0.997012 | 0.991435 | 1.000000 | 0.865165 | 0.778758 | 0.799725 | 0.955038 | 0.476886 | 0.060000 | 2.500000 |
| widegood_mediumguard_002 | 0.997012 | 0.991435 | 1.000000 | 0.865165 | 0.778748 | 0.799176 | 0.955490 | 0.476886 | 0.060000 | 4.500000 |
| widegood_mediumguard_003 | 0.997012 | 0.991435 | 1.000000 | 0.865165 | 0.778743 | 0.798901 | 0.955716 | 0.476886 | 0.060000 | 5.500000 |
| widegood_mediumguard_004 | 0.997012 | 0.991435 | 1.000000 | 0.865165 | 0.778743 | 0.798901 | 0.955716 | 0.476886 | 0.060000 | 6.500000 |
| widegood_mediumguard_001 | 0.997012 | 0.991435 | 1.000000 | 0.865047 | 0.778666 | 0.799176 | 0.955264 | 0.476886 | 0.060000 | 3.500000 |
| widegood_mediumguard_007 | 0.997012 | 0.991435 | 1.000000 | 0.865047 | 0.778612 | 0.793681 | 0.959783 | 0.476886 | 0.060000 | 12.000000 |
| widegood_mediumguard_006 | 0.997012 | 0.991435 | 1.000000 | 0.865047 | 0.778606 | 0.795330 | 0.958427 | 0.476886 | 0.060000 | 10.000000 |
| widegood_mediumguard_005 | 0.997012 | 0.991435 | 1.000000 | 0.864929 | 0.778571 | 0.797802 | 0.956168 | 0.476886 | 0.060000 | 8.000000 |
| widegood_mediumguard_009 | 0.992530 | 0.978587 | 1.000000 | 0.859738 | 0.774639 | 0.782143 | 0.959105 | 0.476886 | 0.080000 | 3.500000 |
| widegood_mediumguard_010 | 0.992530 | 0.978587 | 1.000000 | 0.859738 | 0.774633 | 0.781868 | 0.959331 | 0.476886 | 0.080000 | 4.500000 |
| widegood_mediumguard_012 | 0.992530 | 0.978587 | 1.000000 | 0.859384 | 0.774358 | 0.780495 | 0.959783 | 0.476886 | 0.080000 | 6.500000 |
| widegood_mediumguard_008 | 0.992530 | 0.978587 | 1.000000 | 0.859266 | 0.774324 | 0.782692 | 0.957750 | 0.476886 | 0.080000 | 2.500000 |

## Best Original-Test Macro-F1 Among Node-Viable Guards
| rule | node_acc | node_good | node_medium | acc | macro_f1 | good_recall | medium_recall | bad_recall | medium_guard_qv_max | medium_guard_pc2_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| widegood_mediumguard_000 | 0.997012 | 0.991435 | 1.000000 | 0.865165 | 0.778758 | 0.799725 | 0.955038 | 0.476886 | 0.060000 | 2.500000 |
| widegood_mediumguard_002 | 0.997012 | 0.991435 | 1.000000 | 0.865165 | 0.778748 | 0.799176 | 0.955490 | 0.476886 | 0.060000 | 4.500000 |
| widegood_mediumguard_003 | 0.997012 | 0.991435 | 1.000000 | 0.865165 | 0.778743 | 0.798901 | 0.955716 | 0.476886 | 0.060000 | 5.500000 |
| widegood_mediumguard_004 | 0.997012 | 0.991435 | 1.000000 | 0.865165 | 0.778743 | 0.798901 | 0.955716 | 0.476886 | 0.060000 | 6.500000 |
| widegood_mediumguard_001 | 0.997012 | 0.991435 | 1.000000 | 0.865047 | 0.778666 | 0.799176 | 0.955264 | 0.476886 | 0.060000 | 3.500000 |
| widegood_mediumguard_007 | 0.997012 | 0.991435 | 1.000000 | 0.865047 | 0.778612 | 0.793681 | 0.959783 | 0.476886 | 0.060000 | 12.000000 |
| widegood_mediumguard_006 | 0.997012 | 0.991435 | 1.000000 | 0.865047 | 0.778606 | 0.795330 | 0.958427 | 0.476886 | 0.060000 | 10.000000 |
| widegood_mediumguard_005 | 0.997012 | 0.991435 | 1.000000 | 0.864929 | 0.778571 | 0.797802 | 0.956168 | 0.476886 | 0.060000 | 8.000000 |
| widegood_mediumguard_009 | 0.992530 | 0.978587 | 1.000000 | 0.859738 | 0.774639 | 0.782143 | 0.959105 | 0.476886 | 0.080000 | 3.500000 |
| widegood_mediumguard_010 | 0.992530 | 0.978587 | 1.000000 | 0.859738 | 0.774633 | 0.781868 | 0.959331 | 0.476886 | 0.080000 | 4.500000 |
| widegood_mediumguard_012 | 0.992530 | 0.978587 | 1.000000 | 0.859384 | 0.774358 | 0.780495 | 0.959783 | 0.476886 | 0.080000 | 6.500000 |
| widegood_mediumguard_008 | 0.992530 | 0.978587 | 1.000000 | 0.859266 | 0.774324 | 0.782692 | 0.957750 | 0.476886 | 0.080000 | 2.500000 |

## Interpretation
- If a low-qv/low-PC2 medium guard improves original while node gates stay clean, the remaining high-QRS medium errors are not truly good; they are QRS-spiky but morphologically degraded medium.
- This is still a simple two-part rule: good rescue by PC1/QRS prominence, medium protection by QRS visibility/PC2.
