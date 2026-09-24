# 问题 2：掩码二阶逐元素残差融合单变量设计

## 目标

从 A（`q2-valid-no-train-missingness`）独立派生候选 K，在 Attachment 2 的 `train=3395`、`valid=728` 上以固定 `seed=20260924` 检验 clean valid macro-F1 是否达到 `0.6212527658`。训练、早停、候选选择、报告和代码测试都不得读取、索引或验证 Attachment 2 `test`。

## 选择依据

J 的 text-anchor residual 将 pooled text 的一阶全局残差叠加到融合表示后，clean macro-F1 从 A 的 `0.6165677806` 降至 `0.6128403025`，并使真 Negative/Positive 更常被判为 Neutral；虽然文本缺失场景改善，但这不能替代 clean 主门槛。因此 K 不重试 J 的文本锚点、其缩放或 coverage，也不采用诊断中提出的 text-query cross-attention，因为它与已淘汰的 MulT-lite 的核心机制重叠。

候选 K 只检验一个尚未隔离的假设：A 的可用三模态投影状态在逐帧凸门控前，是否缺少表示同步跨模态一致/冲突的二阶项。[TFN](https://aclanthology.org/D17-1115/) 将一阶、双模态和三模态动态区分建模；[MFB](https://openaccess.thecvf.com/content_iccv_2017/html/Yu_Multi-Modal_Factorized_Bilinear_ICCV_2017_paper.html) 的动机是线性融合不能表达跨模态关联，而完整双线性计算昂贵。K 只采用它们的二阶交互动机，以无参数的对角 Hadamard 项作受限近似，不声称复现 TFN 或 MFB。带新增因子和固定 rank 的 LMF 留作后备，因为其参数、初始化与 rank 选择会破坏当前单次小样本比较的简洁性。

## 接口与数据流

新增 `fusion_variant="pairwise_hadamard_residual"`。候选 K 与 A 仅此字段不同；对齐版输入、冻结 BERT、flat 分类头、identity adapter、关闭训练合成缺失、优化器、学习率 `0.001`、epoch、batch size、损失权重、类别权重、随机种子和 CUDA 均不变。

在当前代码已经完成原始值屏蔽与投影 bias 清零后，令 `z_t`、`z_a`、`z_v` 为三个 `states[B,L,D]`，`p` 为 `masks.temporal[B,L]`，`m_t`、`m_a`、`m_v` 为既有 availability mask。保留 A 的原 gated 表示 `g_A`，再定义：

```text
r_ta = p * m_t * m_a * (z_t * z_a)
r_tv = p * m_t * m_v * (z_t * z_v)
r_av = p * m_a * m_v * (z_a * z_v)
n    = p * (m_t*m_a + m_t*m_v + m_a*m_v)
r_K  = (r_ta + r_tv + r_av) / clamp_min(n, 1)
g_K  = g_A + r_K
```

`n` 是每个样本、每个时间槽实际可用的模态对数，不是可调系数。后续原有 shared temporal encoder、pooling、分类/回归头、loss、checkpoint 规则和输出接口完全不变。

## 不变量

- K 不新建 `nn.Module`、参数、state-dict key、optimizer 参数组、BERT 调用、loss、超参数或缺失模拟规则；候选与 A 的参数键和参数数必须相同。
- 输入不可用时，原始值和 projection bias 已在 `states` 前清零；参与该模态的 pair residual 必严格为零。向任意不可用 raw text/audio/vision 槽填入有限大值，公共输出不能变化。
- 单模态可用或没有模态对时，`r_K=0`，因此 K 与同权重 A 的 logits、score、gates 与 temporal attention 必逐元素相同。所有模态可用时恰平均三个 pair；恰两模态可用时只保留对应一个 pair。
- `p=0` 的 padding 槽 residual 必为零，不改变已有有效长度、attention mask 或 pooling 分母；K 不将全零段重新解释为缺失语义。
- 训练和选择只能调用 Attachment 2 的 train/valid loader。保存后的 valid 重建必须以 `strict=True` 恢复 `pairwise_hadamard_residual`；Attachment 3 仅生成 30 条无标签预测。

## 测试与一次性判定

先为 helper 写手算 fixture，覆盖 0、1、3 个可用 pair 和 padding；再验证 raw-value 不可用性、无 pair 的精确 gated 回退、输出有限、state-dict/参数数不变，以及 fake archive 的 train/valid-only 训练和 strict saved-valid 重建。每个 fake archive 将 Attachment 2 `test` 设为不可访问哨兵。

审查和全套测试后，以新的忽略配置运行一次 `--check`，必须得到 `3395/728/30` 且目标输出目录不存在。随后只训练一次，比较 A/K 的归一化清单，唯一差异必须为 `fusion_variant: gated -> pairwise_hadamard_residual`。接受条件只有 clean valid macro-F1 `>=0.6212527658`；若未达到，永久淘汰该精确 K 实现，不调 pair 归一化、混合系数、loss、宽度、rank、checkpoint、学习率或 test。
