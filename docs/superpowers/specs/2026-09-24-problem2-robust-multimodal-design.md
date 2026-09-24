# 问题 2：局部模态缺失下的鲁棒情感预测设计

## 目标与边界

本设计实现题目问题 2：仅用附件 2 的训练集学习一个对连续局部模态缺失具有鲁棒性的三模态模型，在附件 2 验证集报告情感极性 Accuracy、macro-F1 和情感强度 MAE、Pearson，并对附件 3 的全部 30 个对齐版本样本生成无标签最终预测。

本阶段不训练或调用附件 1 的自建特征，不读取附件 4，不用附件 3 调参或计算指标，不下载模型，也不引入赛题外的情感数据。任务 1 的产物和代码保持不变。

## 已验证的数据契约

选择对齐版本，因其固定为 50 个位置并与附件 3 对齐版本一一对应。

| 证据 | 结论 |
| --- | --- |
| 附件 2 `aligned_50.pkl` | `train/valid/test` 分别有 3395/728/727 条；`audio[N,50,74]`、`vision[N,50,35]`、`text[N,50,768]`、`text_bert[N,3,50]`。 |
| 类别与强度交叉核验 | `0=Negative` 且强度全小于 0；`1=Neutral` 且强度恒为 0；`2=Positive` 且强度全大于 0。 |
| 附件 3 对齐版 | 每个文件是 `{"test": {"audio", "vision", "text_bert"}}`，单样本形状为 `[1,50,74]`、`[1,50,35]`、`[1,3,50]`；没有 `text` 字段。 |
| 共同文本接口 | 附件 2 `text_bert` 是 `int64`，附件 3 是数值等价的 `float32`；二者均为 BERT 的 token-id、attention-mask、token-type-id 三行。用本地 `bert-base-uncased` 重编码附件 2 时，与其 `text` 在有效 token 上的 MAE 为约 `6.1e-7`，平均余弦为 1.0。 |

因此运行时唯一文本入口是 `text_bert`。加载器将拒绝非整数值或错误形状的附件 3 token 数据，再安全转换为 `int64`；不会把浮点特征静默截断成 token ID。

## 数据流与可用性规则

`SevenZipArchive` 逐成员流式读取 pickle，不解压完整归档。附件 2 一次读入对齐训练/验证划分，附件 3 逐文件读取，避免创建原始数据副本。

令位置为 `t=1..50`。文本基础可用性来自 `text_bert[1,t]` 的 attention mask。音频和视觉的观测可用性分别以该位置特征向量是否全零判断。总体时间位置掩码为三者的并集：

`temporal[t] = text_available[t] OR audio_nonzero[t] OR vision_nonzero[t]`。

这一定义将题面指定的“全零连续区间”保留为不可用证据。零值在没有题面缺失语义的其他数据接口中不作这种推断；对齐附件 2 的尾部填充由同一规则和 BERT attention mask 排除。若多个模态同一位置同时全零，该位置没有可用证据，模型不会把它伪造成有效观测。

## 模型

文本分支使用本地、冻结、评估模式的 `bert-base-uncased`，将三行 `text_bert` 变为 `H_text[50,768]`。冻结 BERT 是开源基础特征提取器，不接触外部情感标签或在线下载。文本、音频、视觉各自经线性投影、LayerNorm 和 GELU 变成 128 维状态。

每个位置的门控网络接收三路投影和各自可用性指示，并对不可用模态施加 `-inf` 掩码后在可用模态之间 softmax。门控加权的融合状态进入两层、四头的时间 Transformer；`temporal` 作为 key-padding mask。掩码注意力池化得到片段表示，连接可用模态比例后送入两个头：

- 分类头输出 3 个 logits，对应 Negative、Neutral、Positive；
- 回归头经 `3*tanh(.)` 输出 `[-3,3]` 的强度。

优化目标为 `cross_entropy(class_logits, class_label) + 0.5 * smooth_l1(score, regression_label)`。训练集类别权重按训练标签倒频率归一化，避免将验证/附件 3 信息用于权重选择。模型单独输出分类和回归，不以其中一者反推另一者。

## 训练与缺失增强

音频和视觉只用训练划分的、基础可用位置计算均值和标准差；统计量写进运行清单。每个训练 batch 以固定种子生成局部连续块损失：随机选 1 或 2 种模态、在各样本有效位置内选连续区间、覆盖 10% 至 50% 的有效位置。被选区间只置该模态的模型输入和可用掩码为不可用，标签、原始数组与自然 padding 保持不变。

该增强不是对附件 3 缺失位置的猜测，也不改变附件 2 原始输入。随机数生成器、种子、epoch 和增强参数记录在 `run_manifest.json`，使单次训练可复现。

## 验证与最终推理

验证集首先在干净输入上计算 Accuracy、macro-F1、MAE、Pearson。随后固定地评估 27 个受控场景：目标模态为 text/audio/vision，位置为 beginning/middle/end，时长为 10%/30%/50%。每个场景只对可用的连续位置构造块掩码，输出一张有完整场景列与四项指标的 CSV；没有隐藏标签的附件 3 不出现在这张表中。

选择验证集 clean `macro-F1` 为主、`MAE` 为并列约束的最佳 epoch。完成后在全部 30 个附件 3 对齐 pickle 上推理，生成：

- `attachment3_predictions.csv`：`sample_id`、来源文件、`polarity_class`、`polarity_label`、`sentiment_intensity`；
- `attachment3_missingness.csv`：按模态记录由全零证据得到的前导、尾随、内部连续段摘要；
- `metrics.json`、`validation_scenarios.csv`、`audit_report.md`、`run_manifest.json`、`model.pt`。

所有上述产物写入一个此前不存在的、Git 忽略的目录。`audit_report.md` 明确区分验证指标与附件 3 无标签预测，并说明全零规则只代表题面定义的不可用证据。

## 故障处理与测试策略

运行前检查归档完整性、成员存在、本地 BERT 资产和空输出目录。出现错误时不创建最终输出目录。加载器拒绝缺字段、错误时间/特征维度、非整数 token、非有限数或训练/验证标签不在约定范围的数据。

单元测试覆盖：共同文本接口的 float-token 规范化/拒绝路径、附件 3 成员排序与样本 ID、掩码和连续块边界、训练统计量仅来自训练数据、模型的掩码门控与两个头、全部 27 个验证场景的确定性覆盖，以及预测 CSV 的标签映射与行数。端到端测试使用小型普通 ZIP 和测试替身 token encoder；真实归档训练为显式命令，不作为默认测试。
