# 问题 2：文本锚定残差融合单变量设计

## 目标

从 A（`q2-valid-no-train-missingness`）独立派生候选 J，在 Attachment 2 的 `train=3395`、`valid=728` 上以固定 `seed=20260924` 检验 clean valid macro-F1 是否达到 `0.6212527658`。训练、早停、候选选择、报告和代码测试都不得读取、索引或验证 Attachment 2 `test`。

## 证据与假设

A 的 clean macro-F1 为 `0.6165677806`，仅差 `0.0046849852` 达到门槛；冻结文本输出 adapter F 将其提高到 `0.6180331930`，是当前唯一正向的结构信号。A/F 的音频或视觉遮蔽 9 场景平均 F1 分别约为 `0.615/0.613` 与 `0.617/0.614`，接近 clean；text 遮蔽则降至约 `0.583/0.582`。这说明语言是现有接口中的锚点，而非继续增加非语言交叉注意力、专家或分类后处理的证据。

[Li and Chen (2020)](https://aclanthology.org/2020.ccl-1.101/) 在 MOSI/MOSEI 中将语言作为最终联合表示的主干，令非语言模态提供辅助。候选 J 将该想法限制为当前小样本、冻结 BERT 的可审计实现：保留 A 的完整门控融合路径，额外并行构建一个只读文本的共享参数锚点，而不复现需要额外预测损失或预训练的复杂文本中心模型。

## 接口与数据流

新增受严格验证的 `fusion_variant="text_anchor_residual"`。候选 J 仅将 A 的 `fusion_variant` 从 `gated` 改为该值；对齐版输入、冻结 BERT、flat 分类头、identity adapter、训练合成缺失关闭、优化器、学习率 `0.001`、epoch、batch size、损失权重、类别权重、随机种子和 CUDA 均不变。

模型先执行原有的 availability-masked gated fusion，得到 `h_fused`。然后对已掩码的文本投影状态单独取 `text_temporal = temporal & text_available`，复用**同一** `temporal_encoder` 与 `pool_attention` 只编码文本可用位置，得到 `pooled_text`。每个样本的文本锚点为：

```text
h_text = concat(pooled_text, [text_coverage, 0, 0])
h = h_fused + h_text
```

其中 `text_coverage = sum(text_temporal) / sum(temporal)`。若样本没有可用文本位置，编码器只处理有文本的行，缺失行的 `pooled_text`、coverage 和整个 `h_text` 均严格为零；因此 text-anchor 路径不读取 audio/vision 原始值，也不会让不可用 text 数值泄漏。分类、回归、gate、时间注意力、所有原有损失和输出形状保持不变。

## 保存、边界与测试

`fusion_variant` 已有配置/训练清单/保存 valid 恢复边界；只需将新值列为可选变体。历史清单缺少该字段时仍默认为 `gated`。候选 J 清单须显式记录 `text_anchor_residual`，而保存的 checkpoint 必须以 `strict=True` 重建同一结构。

测试必须证明：新变体保持输出有限和形状；用大数替换不可用模态原始值不改变输出；在 text 全部不可用时，以同一权重构造的 `gated` 与 `text_anchor_residual` 输出完全一致；假归档 `run_q2` 与保存权重 valid 重建都在 `TrainValidPayloadWithInaccessibleTest` 下成功，输出 30 条附件 3 预测并从不访问 test。真实运行前必须全套测试和 `train-q2 --check` 成功，预检计数必须为 `3395/728/30`。

## 一次性运行与判定

完成实现与两阶段审查后，以新的忽略配置 `q2-text-anchor-residual.toml` 运行一次预检，再只训练一次到 `/home/administrator/MyItem/E/artifacts/q2-valid-text-anchor-residual/`。比较 A 时把其历史缺省字段规范化为 `polarity_consistency_loss_weight=0.0`、`fusion_variant="gated"`、`text_adapter_variant="identity"` 和 `classification_variant="flat"`；唯一训练字段差异必须为 `fusion_variant: gated -> text_anchor_residual`。

接受条件只有 clean valid macro-F1 `>=0.6212527658`。无论结果均记录四项 clean 指标、逐类 F1、27 场景摘要和输入/清单差异。若未达标，永久淘汰该精确文本锚定残差，不调缩放、coverage、分支权重、文本编码层、损失、分类权重、checkpoint 或学习率。
