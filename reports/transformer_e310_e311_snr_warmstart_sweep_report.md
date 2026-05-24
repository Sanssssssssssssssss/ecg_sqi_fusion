# E3.10/E3.11 SNR Warm-Start Sweep

Goal: test whether D1 warm-start plus explicit SNR learning can rescue the visual-SNR datasets without changing the model architecture.

| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- |
| D1 E3.9a reference | 0.9465 | 0.9153 | 0.9465 | 0.9777 | `[[616, 53, 4], [22, 637, 14], [2, 13, 658]]` |
| E3.10 M0 baseline | 0.9271 | 0.9262 | 0.9020 | 0.9529 | `[[728, 57, 1], [50, 709, 27], [4, 33, 749]]` |
| E3.10 M1 D1 warm-start | 0.9334 | 0.9542 | 0.8906 | 0.9555 | `[[750, 32, 4], [66, 700, 20], [6, 29, 751]]` |
| E3.10 M2 warm-start + SNR head | 0.9402 | 0.9491 | 0.9135 | 0.9580 | `[[746, 35, 5], [48, 718, 20], [7, 26, 753]]` |
| E3.10 M3 warm-start + SNR head + low denoise | 0.9372 | 0.9389 | 0.9173 | 0.9555 | `[[738, 43, 5], [45, 721, 20], [7, 28, 751]]` |
| E3.10 M4 M3 + noise type head | 0.9364 | 0.9427 | 0.9173 | 0.9491 | `[[741, 43, 2], [50, 721, 15], [7, 33, 746]]` |
| E3.11 M0 baseline | 0.8810 | 0.9004 | 0.8072 | 0.9353 | `[[696, 63, 14], [104, 624, 45], [9, 41, 723]]` |
| E3.11 previous best tune | 0.8698 | 0.8409 | 0.8292 | 0.9392 | `[[650, 113, 10], [75, 641, 57], [3, 44, 726]]` |
| E3.11 M1 D1 warm-start | 0.9000 | 0.9017 | 0.8629 | 0.9353 | `[[697, 65, 11], [61, 667, 45], [5, 45, 723]]` |
| E3.11 M2 warm-start + SNR head | 0.8931 | 0.8887 | 0.8551 | 0.9353 | `[[687, 76, 10], [60, 661, 52], [5, 45, 723]]` |
| E3.11 M3 warm-start + SNR head + low denoise | 0.8883 | 0.8926 | 0.8706 | 0.9017 | `[[690, 77, 6], [71, 673, 29], [5, 71, 697]]` |
| E3.11 M4 M3 + noise type head | 0.8922 | 0.8862 | 0.8758 | 0.9146 | `[[685, 79, 9], [56, 677, 40], [5, 61, 707]]` |

## Best New Runs

- E3.10 best new run: `E3.10 M2 warm-start + SNR head` = `0.9402`
- E3.11 best new run: `E3.11 M1 D1 warm-start` = `0.9000`

Success criteria:

- E3.11 `>=0.90`: rescued enough for further analysis
- E3.11 `>=0.93`: usable visual benchmark
- E3.11 `>=0.94`: strong visual benchmark
- E3.10 `>=0.94`: candidate visual version if E3.11 remains too severe
