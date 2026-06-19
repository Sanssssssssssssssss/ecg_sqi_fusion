# Clean BUT Data + Interpretable Waveform Transformer Experiment Plan

## 结论先行

审阅意见是对的：下一步不应该继续盲目放大模型、继续扫 class weight、继续堆 synthetic boundary。当前真正要修的是：

1. **验证协议**：current test 几乎是 record `111001` 压力测试，val 不是 test 的可靠代理。
2. **数据窗口协议**：当前 10s 窗口虽然都在长 consensus label segment 内，但固定 10s 截取仍可能错过/稀释 contact、baseline、QRS dropout 等局部事件证据。
3. **输入证据链**：现有 robust3/per-window normalization 可能丢掉 absolute amplitude/contact/baseline 证据。
4. **模型决策链**：Transformer 已经能恢复不少 SQI，但分类头没有稳定、显式地使用这些 SQI 证据。

新跑的 clean-window audit 说明：短标签不是主因；短于 10s 的 segment 已经被 protocol 丢掉。但 outlier_low_confidence 是主因之一，尤其是 bad outlier。它不是“脏标签短段”，而是 record-specific morphology stress。

## 新审计结果

审计脚本：

`outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/design_clean_but_window_experiments.py`

主报告：

`reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/clean_but_window_policy/clean_but_window_policy_experiment_design.md`

关键数字：

| policy | n | good | medium | bad | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| current_all_10s | 32956 | 17043 | 10628 | 5285 | 当前全量 10s |
| margin_ge_2s | 29959 | 15381 | 9411 | 5167 | 离标签段边界 >=2s，仍保留 90.9% |
| margin_ge_5s | 29410 | 15042 | 9212 | 5156 | 离边界 >=5s，仍保留 89.2% |
| margin_ge_2s_drop_outlier | 21914 | 11458 | 6374 | 4082 | 去掉 outlier_low_confidence 后明显变小 |
| clean_core_plus_overlap_margin2 | 17823 | 11458 | 6365 | 0 | good/medium learnable body |
| clean_core_only_margin2 | 4944 | 3196 | 1748 | 0 | 过于干净，只能做 sanity |

如果从原始 consensus segment 中心重新生成窗口，而不是沿用当前固定窗口：

| regeneration policy | candidate windows | good | medium | bad |
| --- | ---: | ---: | ---: | ---: |
| center_one_per_segment_margin5 | 1732 | 1000 | 679 | 53 |
| center_stride10_margin5 | 30320 | 15574 | 9561 | 5185 |
| center_stride5_margin5 | 59730 | 30616 | 18773 | 10341 |

这说明我们有足够空间构造新的干净协议。最值得做的是 `center_stride10_margin5` 或 `center_stride5_margin5`：它们保留足够 bad，同时避免窗口贴着标签段边缘。

## 实验 0：先固定验证协议

目的：避免再用“容易 val + 111001 hard test”误导模型设计。

设计：

- 保留 current test 作为 `legacy_record111_stress`，只做 stress report。
- 新增 `grouped_record_cv`：按 record/subject 分组，class、original_region、bad outlier 比例尽量平衡。
- 新增两个压力 fold：
  - `leave_105001_out`：检查 bad 是否只记住 105001。
  - `leave_100001_out`：检查 good/medium 是否只记住 100001/105001。
- 禁止 `train_scope=trainval` 后仍把同一个 val 当独立验证。
- bad threshold / calibration 只能用 out-of-fold logits 或独立 calibration fold。

主指标不再只看 window acc：

`score = 0.45 * record_macro_F1 + 0.25 * class_record_balanced_recall + 0.20 * worst_major_record_recall - 0.10 * ECE`

同时报告 full test、clean body、bad outlier stress。

## 实验 1：Clean 10s learnable body

目的：验证模型能不能在“足够干净、可学习”的 BUT 子集上学到稳定 SQI/形态规则。

数据：

- `margin_ge_2s_drop_outlier`
- `margin_ge_5s_drop_outlier`
- `clean_core_plus_overlap_margin2`

训练：

- 只用 waveform-only Transformer。
- 不使用 PCA/KNN/region_confidence 作为正式输入或正式 aux target。
- 只用可由 waveform 稳定计算的 targets：
  - 7 SQI，尤其 `sqi_basSQI`
  - QRS count / RR median-IQR / detector agreement
  - qrs_visibility / qrs_prominence / qrs_band_ratio
  - baseline_step / flatline/contact_loss
  - non-QRS detail / 15-30 Hz / 30-45 Hz / derivative tails

验收：

- clean body >=0.90 是基础线。
- 如果 clean body 都不到 0.90，优先修输入和模型，不继续碰 outlier。

## 实验 2：重新生成 centered clean windows

目的：验证当前固定 10s 截法是否打断了事件证据。

数据协议：

- `center_stride10_margin5`
- `center_stride5_margin5`

核心变化：

- 从原始 consensus segment 中心/内部重新采样，不从 segment start 固定切。
- 每个 10s 窗口左右都至少离标签段边界 5s。
- 允许同一长 segment 生成多个干净窗口，但训练 sampler 必须限制同 record / 同 segment 过采样。

验收：

- 如果 centered window 明显改善 record-balanced fold，说明 fixed-window protocol 是核心问题之一。
- 如果不改善，问题更可能是 record morphology shift 和输入 evidence loss。

## 实验 3：Variable-length / multi-crop segment model

目的：不再强行把长标签段压成单个 10s 决策。

两种实现：

1. **MIL multi-crop**：一个 segment 内采多个 clean 10s crops，segment-level label 聚合；训练时 crop-level + segment-level loss。
2. **Variable-length masked Transformer**：输入 10-30s 或更长片段，带 mask；局部 event token + top-k/noisy-OR pooling。

优点：

- contact loss / baseline drift / QRS dropout 可能是局部事件，单个 10s crop 很容易错过。
- segment-level 聚合更接近人工标注逻辑。

验收：

- record-macro F1 提升；
- bad stress recall 提升时 medium false-bad 不爆炸。

## 实验 4：修输入链路，不先改大模型

目的：保留 absolute/contact evidence。

现有问题：

- per-window median/RMS 归一化可能抹掉 amplitude collapse、absolute gain、slow contact loss。
- augmentation 发生在 robust3 派生通道之后，会破坏 derivative/baseline 通道一致性。

新输入：

1. physical/global-normalized waveform
2. per-window robust-normalized waveform
3. derivative，必须从增强后的 waveform 重新计算
4. long baseline / lowpass trend，建议 >=2s trend
5. local envelope/log-RMS，250ms 和 1s 两个尺度
6. optional validity/flatline mask

augmentation 顺序：

`raw waveform -> physical augmentation -> recompute derivative/baseline/envelope -> model input`

禁止：

- circular roll
- 对 derivative/baseline 派生通道直接施加不一致扰动
- 把 10s 归一化为 0-1 后误把周期数当 Hz

## 实验 5：显式 SQI/query/层级分类

目的：解决“模型能预测 SQI，但分类头没稳定使用 SQI”的问题。

结构：

- QRS/RR query：QRS count、RR median/IQR、detector agreement、missing/spurious QRS。
- Baseline/contact query：baseline_step、slow drift、contact-loss interval、flatline/clipping。
- Detail/frequency query：qrs-band ratio、non-QRS detail、15-30/30-45 Hz。
- Bad-stress query：segment-level bad evidence，用 top-k / LogSumExp / noisy-OR pooling。
- Class query：最终分类。

分类方式：

- Head A: bad vs non-bad
- Head B: good vs medium conditioned on non-bad

概率：

`P(bad)=p_bad`

`P(good)=(1-p_bad)*p_good_given_nonbad`

`P(medium)=(1-p_bad)*(1-p_good_given_nonbad)`

loss：

- hierarchical CE/focal
- good/medium boundary loss
- bad specificity loss：显式惩罚 medium -> bad
- intrinsic SQI aux loss，小权重开始
- feature consistency loss

不要一开始给很大的 aux weight。当前证据说明强 aux 会恢复特征，但也可能破坏分类几何。

## 实验 6：Outlier 不删除，分层处理

目的：既承认 clean learnable body，也不逃避 full BUT。

分桶：

- `clean_learnable_body`: margin clean + clean_core/good_medium_overlap。
- `boundary_conflict`: good/medium overlap + low confidence。
- `bad_core`: near_bad_boundary / right-island。
- `bad_outlier_stress`: record-specific outlier bad。

训练策略：

1. 先让模型在 clean_learnable_body 学到可解释机制。
2. 再加入 controlled outlier curriculum。
3. extreme outlier 先作为 report-only stress。

报告必须同时给：

- full BUT
- clean body
- boundary conflict
- bad core
- bad outlier stress

这样不会变成“删数据刷分”，而是清楚地说明哪些是模型已学会的 ECG quality mechanism，哪些是当前 domain-stress。

## 推荐立即执行顺序

1. 用本次脚本输出的 `margin_ge_2s_drop_outlier` 和 `clean_core_plus_overlap_margin2` 先做 clean body Transformer sanity。
2. 生成 `center_stride10_margin5` 新协议，不再沿用 current fixed segment-start windows。
3. 实现 dual-view 5/6-channel input，先复用现有 Transformer 宽度，不加大模型。
4. 实现 intrinsic-only aux targets，移除 PCA/KNN/region geometry targets。
5. 实现 hierarchical bad/nonbad + good/medium head。
6. 用 grouped record CV + legacy 111001 stress 双轨汇报。

这套路径比继续推 N7180/N7200 或继续扫 class weight 更合理，也更符合“可解释、能讲清楚、模型确实从 waveform 学质量机制”的目标。
