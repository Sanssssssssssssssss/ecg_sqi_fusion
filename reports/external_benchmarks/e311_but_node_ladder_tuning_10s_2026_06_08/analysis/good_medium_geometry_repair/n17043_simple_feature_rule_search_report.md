# N17043 Simple Feature Rule Search

Train/val-only feature rules; original BUT not used. Goal: see whether a small set of ECG geometry/SQI features can explain a large boundary chunk without adding another UFormer data block.

## Best All-Scope Rules

- dtree_depth8_leaf10: all acc=0.9897, G/M/B=0.993/0.980/1.000, complexity=107
- dtree_depth5_leaf10: all acc=0.9779, G/M/B=0.987/0.955/1.000, complexity=33
- dtree_depth5_leaf50: all acc=0.9767, G/M/B=0.987/0.951/1.000, complexity=31
- dtree_depth6_leaf50: all acc=0.9763, G/M/B=0.985/0.954/1.000, complexity=45
- dtree_depth8_leaf50: all acc=0.9763, G/M/B=0.985/0.954/1.000, complexity=65
- dtree_depth6_leaf10: all acc=0.9732, G/M/B=0.989/0.937/1.000, complexity=59
- dtree_depth4_leaf10: all acc=0.9730, G/M/B=0.976/0.959/1.000, complexity=17
- dtree_depth4_leaf50: all acc=0.9730, G/M/B=0.976/0.959/1.000, complexity=17
- dtree_depth3_leaf10: all acc=0.9713, G/M/B=0.978/0.949/1.000, complexity=9
- dtree_depth3_leaf50: all acc=0.9713, G/M/B=0.978/0.949/1.000, complexity=9

## Best Test-Scope Rules

- dtree_depth8_leaf10: test acc=0.9765, G/M/B=0.991/0.964/1.000, complexity=107
- dtree_depth4_leaf10: test acc=0.9481, G/M/B=0.963/0.935/1.000, complexity=17
- dtree_depth4_leaf50: test acc=0.9481, G/M/B=0.963/0.935/1.000, complexity=17
- dtree_depth5_leaf10: test acc=0.9458, G/M/B=0.979/0.917/1.000, complexity=33
- dtree_depth5_leaf50: test acc=0.9451, G/M/B=0.980/0.915/1.000, complexity=31
- dtree_depth6_leaf50: test acc=0.9443, G/M/B=0.974/0.918/1.000, complexity=45
- dtree_depth8_leaf50: test acc=0.9443, G/M/B=0.974/0.918/1.000, complexity=65
- dtree_depth3_leaf10: test acc=0.9428, G/M/B=0.969/0.920/1.000, complexity=9
- dtree_depth3_leaf50: test acc=0.9428, G/M/B=0.969/0.920/1.000, complexity=9
- dtree_depth3_leaf150: test acc=0.9428, G/M/B=0.969/0.920/1.000, complexity=9

## Interpretation

- Simple feature rules can recover bad using high PC1, but they still do not solve the full good/medium ambiguous shell.
- The best ordinary checkpoint plus high-confidence geometry gate is currently the cleanest practical path: full target for stress analysis, retained subset for learnable/reportable diagnostic.
- If a single UFormer checkpoint is mandatory, the next real change should be morphology-level generation, not more row picking from the same PTB pool.
