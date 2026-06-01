# E3.11f Loss Conflict Analysis Brief

请作为一个独立分析聊天/子代理，分析当前 E3.11f denoise-before-classifier + SQI 实验，并给出下一轮实验方向。只做分析和实验设计，不改代码、不启动训练。

## 你必须先看的文件

主实验目录：

- `E:\GPTProject2\ecg\outputs\experiment\e311_sqi_denoise_classifier_grid`

必须读取：

- `conflict_mtl_state.json`
- `conflict_mtl_summary.jsonl`
- `reports\conflict_mtl\CONFLICT_MTL_REPORT.md`
- `full_joint_train.py`
- `adaptive_conflict_mtl_grid.py`
- 每个 top run 的：
  - `train_log.json`
  - `grad_diag.jsonl`
  - `test_report.json`
  - `denoise_eval\denoise_metrics.json`

为了理解前一轮 baseline，还需要读取：

- `focused_2h_summary.jsonl`
- `reports\focused_2h\FOCUSED_2H_REPORT.md`
- `adaptive_full_joint_sqi_2h.py`

为了理解数据和 denoise baseline，还需要知道：

- 数据族：`E:\GPTProject2\ecg\outputs\experiment\e311_morph_denoise_gap5_7_grid\data\med6p25_badgap7_badcm0p75`
- 当前任务使用完整 split：train `10935` / val `2184` / test `2202`
- 当前是 sweep 数据集，不是 original E3.11f 回测。

## 当前架构，不要误解

- 输入是 noisy Lead-I ECG。
- 先过 denoiser：当前满意的路线是 `residual_unet`，预测 noise/residual，然后 `denoise = noisy - noise_scale * noise_hat`。
- 再把 denoised ECG 喂给 Transformer classifier。
- 分类头可选 SQI residual：
  - `gated SQI`：当前主候选
  - `delta SQI`：bad recall 对照
- SQI features 只能使用推理时可获得内容：noisy/denoised/residual stats、base logits/prob、predicted snr/local stats 等。
- 不允许使用 clean、true mask、true morph score、label leakage 作为分类输入。

## 当前最重要结论

这轮不是单纯追 acc，而是判断多任务 loss 是否打架，以及怎样缓解。

15 个 full-data conflict MTL run 已跑完，0 failed。

当前最好的主候选：

- run: `conflict_A_ce_primary_pcgrad_gated_sc0p035_e8_fa64d045`
- strategy: `ce_primary_pcgrad`
- anchor: A, gated SQI
- acc: `0.9654859218891917`
- good/medium/bad recall: `0.9536784741 / 0.9536784741 / 0.9891008174`
- denoise_score: `2.8959095627`

有价值的 bad-special 对照：

- run: `conflict_C_ce_primary_pcgrad_delta_sc0p0175_e8_94091fdb`
- strategy: `ce_primary_pcgrad`
- anchor: C, delta SQI
- acc: `0.9609445958`
- good/medium/bad recall: `0.9373297003 / 0.9550408719 / 0.9904632153`
- denoise_score: `2.9431493467`
- 结论：delta SQI 能保 bad，但牺牲 good 和总 acc，不适合作主候选。

## 当前冲突证据

`grad_diag.jsonl` 里记录的 `grad_denoiser_denoise_cos_mean` 是 raw CE-vs-denoise 梯度 cosine，未经过 PCGrad 投影。

固定 loss 下：

- Anchor A fixed: raw CE vs denoise cosine 约 `-0.224`
- Anchor B fixed: raw CE vs denoise cosine 约 `-0.236`
- Anchor C fixed: raw CE vs denoise cosine 约 `-0.392`

且 denoise 梯度 norm ratio 经常是 CE 的几十倍甚至上百倍。这说明任务本身确实冲突，不是小波动。

特别注意：PCGrad 结果里的 `grad_denoise_cos` 仍然是 raw conflict，不代表 PCGrad 没生效。当前代码还没有记录 projected/applied gradient cosine，这是下一轮必须补的诊断。

## 已尝试策略与初步判断

- `fixed`: 作为对照，raw conflict 明显，整体不如 PCGrad。
- `ce_primary_pcgrad`: 当前最佳方向。CE 梯度原样保留，只对 denoise 梯度在 denoiser 参数上做 conflict projection。效果最好，但还缺 applied gradient 诊断和 norm cap。
- `bounded_gradnorm`: 当前实现偏 loss-level/近似，不是真正 applied gradient cap。结果没有胜出。
- `bounded_uncertainty`: 有时 raw cosine 稍缓，但 aux norm 仍大，bad recall 不理想。
- `alternating`: 不好，分类和 denoise 都没有优势。
- `ce_distill_guard`: 当前 distill 太重，distill 梯度和 CE 更同向但分类掉了。可能需要轻量 high-confidence boundary distill，而不是大权重全样本 distill。

## 请重点分析的问题

1. 下一轮如何真正证明 loss 打架被缓解，而不是只是 test acc 偶然变好？
2. `ce_primary_pcgrad` 应该如何升级：
   - 只投影 denoise？
   - 是否也投影 SNR/local/quality 对 classifier encoder 的冲突梯度？
   - 是否需要 applied gradient norm cap？
3. 应该记录哪些新诊断：
   - raw cosine / projected cosine / applied cosine
   - raw aux norm ratio / applied aux norm ratio
   - CE boundary stability，例如 high-confidence teacher KL、margin preservation
   - SQI corrected/harmed 的类别分布
4. 下一轮最小但有信息量的 grid 应该怎么设计？
5. 哪些方案应该直接砍掉，不要继续浪费资源？

## 我希望你输出的格式

请输出一份下一轮实验建议，包含：

1. 一段“当前结论”。
2. 一个推荐主方向，最好给出明确策略名，例如 `ce_primary_pcgrad_cap`。
3. 一个 8-12 run 的下一轮 grid，必须是完整数据训练，不是 cached head shortcut。
4. 每个 run 的关键超参：
   - anchor
   - SQI variant
   - class weights
   - lambda_den / lambda_quality
   - grad cap ratio
   - 是否使用 light distill
   - select_by
5. 必须新增的诊断字段。
6. 晋级/停止规则。
7. 哪些结果如果出现，就说明“loss 打架真的被解决/没解决”。

## 推荐你优先考虑的下一轮方向

我倾向下一轮测试：

- `ce_primary_pcgrad_cap`
  - CE 梯度完全保留。
  - 对 denoise/SNR/local/quality 中和 CE 冲突的部分做 PCGrad。
  - 投影后再做 aux norm cap，例如 `0.05 / 0.10 / 0.20`。
  - 记录 raw/projected/applied 三套梯度诊断。
- `light_boundary_distill`
  - 只在 teacher 高置信且 student margin 被破坏时加小权重 distill。
  - 权重建议先试 `0.05 / 0.10 / 0.20`，不要再用 `0.75`。
- `freeze_or_decay_denoise_after_good_enough`
  - 如果 val denoise_score 已经 `>=2.90` 且 val acc 开始掉，后几 epoch 降低 denoise 权重或冻结 denoiser。

请你不要只给泛泛文献综述，要基于这些本地结果给出具体下一轮可执行实验设计。
