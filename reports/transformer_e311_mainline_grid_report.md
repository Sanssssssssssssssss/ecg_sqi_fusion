# E3.11 Mainline Grid

Mainline decision: E3.11f-style visual data replaces E3.10 as the target line.
E3.10 remains only a historical reference, not an optimization target.

Current historical references:

- E3.10 best single visual: `0.9402`
- E3.10 calibration/ensemble diagnostic: `0.9419`
- Previous E3.11f best single visual: `0.9381`
- Previous E3.11f ensemble diagnostic: `0.9402`

Stage-1 goal:

- find an E3.11f single model at or above `0.94`
- screen whether relaxed morphology or wider SNR makes the task more learnable
- prune any branch that is clearly worse before expanding the grid

## Data Variants

| Variant | Meaning | Class Counts G/M/B | measured SNR Mean G/M/B | smooth Morph Mean G/M/B | global Noise Mean G/M/B | SQI-SVM | SQI-MLP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `e311f_lite_e310_morph` | E3.11f lite SNR + E3.10 morphology | missing |  |  |  |  |  |
| `e311h_lite_relaxed_morph` | E3.11h lite SNR + relaxed morphology | missing |  |  |  |  |  |
| `e311i_wide_relaxed_morph` | E3.11i wide SNR + relaxed morphology | missing |  |  |  |  |  |

## Transformer Results

### e311f_lite_e310_morph

E3.11f lite SNR + E3.10 morphology.

| Rank | Run | Test Acc | Summary |
| ---: | --- | ---: | --- |
|  | pending |  |  |

### e311h_lite_relaxed_morph

E3.11h lite SNR + relaxed morphology.

| Rank | Run | Test Acc | Summary |
| ---: | --- | ---: | --- |
|  | pending |  |  |

### e311i_wide_relaxed_morph

E3.11i wide SNR + relaxed morphology.

| Rank | Run | Test Acc | Summary |
| ---: | --- | ---: | --- |
|  | pending |  |  |

## Current Best

| Rank | Variant | Run | Test Acc | Summary |
| ---: | --- | --- | ---: | --- |
|  | pending | pending |  |  |

## Pruning Rules

- If a variant's first 3-6 runs stay below `0.90`, stop expanding it.
- If relaxed morphology beats E3.11f by `>=0.01`, expand that data branch.
- If wide SNR is high but SQI baselines are also high, keep it as visual/diagnostic rather than the main benchmark.
- If positional embedding helps the top E3.11f runs, keep it in round 2; otherwise remove it.
- If denoise-aware runs lose more than `0.01` versus their paired classifier-only run, remove denoise from the next grid.
