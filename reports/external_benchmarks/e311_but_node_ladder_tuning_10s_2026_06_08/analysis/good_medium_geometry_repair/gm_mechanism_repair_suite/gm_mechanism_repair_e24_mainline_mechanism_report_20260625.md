# E24 GM Mechanism Mainline Report

Date: 2026-06-25

Policy: `ptb_v112_gm_buffered_large_hybrid_s20260741`

Runner: `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_gm_mechanism_repair_suite.py`

## 1. 这轮实验做了什么

这轮不是重新造数据，也不是继续换 encoder，而是在当前固定数据 v112 和固定 waveform-only Conformer 主线上，验证 E24 之后几个更保守的融合/调参是否能继续提升：

- `E24_e6_subtype_fusion_pairrank`: 当前主线，factor-fused GM + unified subtype class fusion + pair ranking。
- `E25_e24_lowalpha`: 降低 subtype-class fusion alpha，从 0.20 降到 0.14。
- `E26_e24_lowlr`: 保持 E24 机制，但学习率从 `1.5e-4` 降到 `1.0e-4`。
- `E27_e24_lowalpha_lowlr`: 同时降低 alpha 和学习率。
- `E28_e24_softpair_lowalpha`: pair ranking 从 0.08 降到 0.04，alpha 改成 0.16。

所有候选均从零训练，3 folds，12 epochs，batch size 128。没有使用旧 checkpoint warm-start。

## 2. 结论

`E24_e6_subtype_fusion_pairrank` 仍然是当前主线。E25-E28 都没有超过 E24。

这说明当前瓶颈不是简单的 alpha/lr/pairrank 权重没有调好，而是更底层的局部证据仍然不够稳定，尤其是：

- `detector_agreement` class-wise recovery 仍弱；
- `template_corr` class-wise recovery 仍弱；
- `contact_loss_win_ratio` 几乎没有稳定学到；
- good/medium 边界仍然存在双向互吃，尤其 medium 被 good 吃掉还没有完全解决。

## 3. 主要结果

| candidate | acc | macro-F1 | good | medium | bad | GM balanced | good->medium | medium->good |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E24 mainline | 0.924382 | 0.908527 | 0.902949 | 0.856647 | 0.972482 | 0.879798 | 76.0 | 116.7 |
| E23 subtype fusion | 0.923968 | 0.907960 | 0.875661 | 0.866162 | 0.977958 | 0.870911 | 95.0 | 92.7 |
| E21 subtype consistency | 0.922726 | 0.906753 | 0.881820 | 0.864165 | 0.974426 | 0.872992 | 92.0 | 100.3 |
| E25 low alpha | 0.922726 | 0.906164 | 0.882048 | 0.853380 | 0.979923 | 0.867714 | 89.7 | 102.3 |
| E26 low lr | 0.921483 | 0.905404 | 0.913846 | 0.843228 | 0.969684 | 0.878537 | 69.7 | 129.3 |
| E27 low alpha + low lr | 0.919082 | 0.902271 | 0.874593 | 0.867868 | 0.967404 | 0.871231 | 101.3 | 105.7 |
| E28 soft pairrank | 0.918833 | 0.901167 | 0.865621 | 0.861965 | 0.974726 | 0.863793 | 107.7 | 102.3 |

Interpretation:

- E24 不是每个单项都最高，但综合最好：总体 acc、macro-F1、GM balanced、bad recall 和 false-bad 控制最均衡。
- E26 的 good recall 更高，但 medium recall 明显掉到 0.843，说明保守学习率让模型更偏 good，不适合作为主线。
- E23 medium 和 bad 更好，但 good 明显低于 E24；如果后续目标是保 medium，可以作为对照，不建议替代主线。
- E25/E27/E28 说明 subtype 融合减弱或 pairrank 变软都会降低整体收益。

## 4. 当前主线模型机制

当前正式主线是 `GMMechanismConformer`，它挂在 `EventFactorizedSQIConformer` 上。推理时仍然只输入 waveform-derived channels，不输入 47-feature 表格，不输入 MLP/tree/rule 输出。

### 4.1 输入

模型输入为当前外部 runner 中统一生成的 waveform-derived view。它包括 raw ECG 及其导数、baseline/detail/filterbank-like 派生通道等波形可计算信息。47 个 SQI/geometry/factor 只作为训练期 teacher target 和诊断指标，不作为 inference input。

### 4.2 主体 encoder

基础 encoder 为 `EventFactorizedSQIConformer`：

- width: 96
- layers: 3
- heads: 4
- dropout: 0.12
- weight decay: 2e-4
- E24 lr: 1.5e-4

结构上保留 high-resolution stem 和 context tokens。high-res 路径负责局部 QRS/contact/flatline/reset/detail 证据；context/Conformer 路径负责 10s 窗口内的节律、持续性和全局质量。

### 4.3 机制 query tokens

模型内部使用 factor/task query tokens 组织 ECG quality 机制：

- `QRS`
- `RR_TEMPLATE`
- `BASELINE`
- `CONTACT_RESET`
- `DETAIL_NOISE`
- `GLOBAL_MORPH`
- `GM_BOUNDARY`
- `BAD_STRESS`

这些 token 的目的不是做黑箱全局 pooling，而是让不同质量机制各自有读出位置。GM 决策主要读 `GM_BOUNDARY`，bad 决策主要读 `BAD_STRESS`。

### 4.4 Local map heads

模型会从 high-res stem 预测局部图：

- QRS event map A/B
- baseline map
- contact mask
- reset mask
- flatline mask
- detail/noise map

`detector_agreement` 不再从任意隐藏维度直接拿，而是由 local QRS event maps 计算得到。这一点是前面修复 factor contract 后的重要约束。

### 4.5 Factor heads

模型训练期预测 waveform-computable factors，例如：

- `qrs_visibility`
- `detector_agreement`
- `baseline_step`
- `flatline_ratio`
- `sqi_basSQI`
- `non_qrs_diff_p95`
- `non_qrs_rms_ratio`
- `qrs_band_ratio`
- `template_corr`
- `amplitude_entropy`
- `contact_loss_win_ratio`

这些 factor 是训练期监督和内部 evidence，不是外部表格输入。

### 4.6 层级三分类概率

三分类不是普通 softmax，而是层级概率：

```text
b = sigmoid(bad_logit)
m = sigmoid(medium_logit)

P(good)   = (1 - b) * (1 - m)
P(medium) = (1 - b) * m
P(bad)    = b
```

这样 bad 与 good/medium 边界分开，good/medium 只在 non-bad 条件下处理。

### 4.7 E24 的 GM factor fusion

E24 的 medium logit 不只读 `GM_BOUNDARY` token，而是：

```text
medium_logit =
    gm_direct_head(GM_BOUNDARY)
  + gm_factor_head([GM_BOUNDARY, decoded_factor_pred, local_stats])
```

其中 `local_stats` 包括：

- QRS top-k confidence
- contact longest run
- flatline longest run
- baseline mean/top-k
- detail top-k
- reset peak count

这就是 E24 比普通 Conformer 更强的核心：good/medium 判断被迫读取 waveform 预测出的局部质量证据，而不是只靠全局 embedding。

### 4.8 Unified quality subtype head

E24 保留统一 subtype 逻辑。模型输出所有 leaf subtype 的 logits，再把 subtype leaf 按 class 汇总成：

```text
P_subtype(good)
P_subtype(medium)
P_subtype(bad)
```

然后和层级三分类概率做 log-space fusion：

```text
log P_final =
  (1 - alpha) * log P_hier
+ alpha       * log P_subtype_class
```

E24 使用 `alpha = 0.20`。E25 把 alpha 降到 0.14 后整体变差，说明 subtype class evidence 是有用的，不能太弱。

### 4.9 Pair ranking

E24 使用 `pairrank_weight = 0.08`。它在可构成同 family good/medium 对照的样本上，要求 medium row 的 medium score 高于 good row。这个 loss 不伪造 pair；pair 不足时只作为已有 pair 的边界约束。

E28 把 pairrank 降到 0.04 后变差，说明这个边界排序约束对当前 v112 数据是正收益。

## 5. 模型目前已经学会了什么

E24 key feature recovery:

| feature | corr all | min class corr | MAE |
| --- | ---: | ---: | ---: |
| baseline_step | 0.9683 | 0.8710 | 0.0120 |
| non_qrs_rms_ratio | 0.9677 | 0.9128 | 0.0394 |
| qrs_visibility | 0.9643 | 0.7175 | 0.0810 |
| sqi_basSQI | 0.9666 | 0.7823 | 0.0239 |
| qrs_band_ratio | 0.9520 | 0.8233 | 0.4992 |
| template_corr | 0.9496 | 0.2756 | 0.0741 |
| detector_agreement | 0.7518 | 0.1336 | 0.2584 |
| contact_loss_win_ratio | 0.2014 | -0.0393 | 0.0054 |

Interpretation:

- `baseline_step`, `non_qrs_rms_ratio`, `qrs_visibility`, `sqi_basSQI`, `qrs_band_ratio` 已经能从 waveform 学到比较强。
- `detector_agreement` all-class 看起来可以，但 class-wise 很弱，说明某些类别里 detector/template 证据仍然没有被稳定恢复。
- `template_corr` all-class 很高，但 min-class 很弱，说明它可能依赖类别分布捷径，不是每一类都真的稳。
- `contact_loss_win_ratio` 基本没学好。这是下一步最明确的结构瓶颈之一。

## 6. 为什么没有到 0.95

当前不是坏在 bad：E24 bad recall 已经约 0.9725。主要短板还是 good/medium：

- good recall: 0.9029
- medium recall: 0.8566
- good -> medium: 76.0
- medium -> good: 116.7

也就是说 medium 被 good 吃掉更多。E26 虽然 good recall 到 0.9138，但 medium 掉到 0.8432，证明简单调学习率会把边界往 good 推，而不是学到更稳的边界。

现阶段最可信的解释是：模型已经能学到大部分全局和低频/幅值类 SQI，但对 detector/template/contact 这类局部、事件级、离散形态证据仍不够稳。因此继续扫 alpha/lr/pairrank 不会自然突破 0.95。

## 7. 下一步建议

建议保留 E24 为主线，不再围绕 alpha/lr/pairrank 小幅扫参。下一步应该专门修局部事件证据：

1. `contact_loss_win_ratio` 专项：
   - 改 local contact/flatline/reset target 质量；
   - 增加 longest-run/event-boundary loss；
   - 输出 contact-positive nonbad 与 bad contact 的可视化面板。

2. `detector_agreement` 专项：
   - 从 dense map 均值进一步改成 event matching / peak count / RR consistency；
   - 对 class-wise recovery 设硬门槛，不只看 all-class corr。

3. `template_corr` 专项：
   - 增加 beat-template token 或 beat prototype consistency loss；
   - 检查 medium->good 错误是否集中在 template_corr 预测过高的 subtype。

4. good/medium 边界：
   - 继续用 E24 的 unified subtype + pairrank 机制；
   - 不要把 E23/E26 直接替换主线，但可以用它们作为两端模型，分析 good-heavy 和 medium-heavy 的差异样本。

## 8. 关键产物位置

Metrics:

- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/candidate_metrics_e21e28_subtype_stability_summary_20260625.csv`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/factor_recovery_e24_e28_key_features_20260625.csv`

Automatic suite report:

- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/gm_mechanism_repair_suite_report_e25e28_stability_20260625.md`

This mechanism report:

- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/gm_mechanism_repair_e24_mainline_mechanism_report_20260625.md`

Runner:

- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_gm_mechanism_repair_suite.py`
