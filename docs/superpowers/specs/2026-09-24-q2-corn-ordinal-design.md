# 问题 2：CORN 有序分类头单变量设计

## 目标

从 A（`q2-valid-no-train-missingness`）独立派生候选 I，在 Attachment 2 的 `train=3395`、`valid=728` 上以固定 `seed=20260924` 检验 clean valid macro-F1 是否达到 `0.6212527658`。训练、早停、模型选择、报告和代码测试都不得读取、索引或验证 Attachment 2 `test`。

## 证据与假设

A 的三类有效样本数为 Negative/Neutral/Positive=`206/184/338`，逐类 F1 为 `0.654155/0.489011/0.706537`。其错误中 `Negative->Positive=55`、`Neutral->Positive=72`、`Positive->Neutral=62`；现有平面三类 softmax 忽略标签 `Negative < Neutral < Positive` 的顺序。H 的低学习率单点同时降低 clean 和缺失场景 F1，不能再用优化器路线解释这些极性跨越。

候选 I 使用 CORN 的条件概率分解：[Shi, Cao and Raschka (2021)](https://arxiv.org/abs/2111.08851) 将有序多分类转为满足 rank consistency 的条件二分类训练，且不依赖特定编码器。它不同于已淘汰的 `polarity_consistency_loss_weight`：后者只以辅助项把独立 softmax 的期望极性与回归输出拉近；CORN 直接替换分类头的概率参数化和分类负对数似然，回归头不参与该分类结构。

## 接口与数据流

新增受严格验证的 `classification_variant`：`flat`（默认、完整兼容）和 `corn`。候选 I 仅将 A 的该字段从 `flat` 改为 `corn`；对齐版输入、冻结 BERT、掩码、`gated` 融合、`identity` adapter、无训练合成缺失、优化器、学习率 `0.001`、epoch、batch size、损失权重、随机种子和 CUDA 均不变。

`corn` 头为现有融合表示输出两个原始条件 logit `r0,r1`：

- `q0=sigmoid(r0)=P(y>0)`；
- `q1=sigmoid(r1)=P(y>1 | y>0)`；
- `p0=1-q0`、`p1=q0*(1-q1)`、`p2=q0*q1`。

这三个概率非负且总和为一，因而可安全转换为 `log(p)` 填入保持 `[B,3]` 形状的 `Q2Output.logits`，供 argmax、指标、缺失场景与附件 3 推理沿用。`Q2Output` 额外携带可选 `ordinal_logits[B,2]`；只有它存在时，训练使用两个条件 BCE 的均值：第 0 阈值使用所有样本的 `y>0`，第 1 阈值只使用 `y>0` 的样本并以 `y>1` 为目标。分类权重不用于 CORN 的二元条件损失，防止将已淘汰的类别权重路线混入该候选；回归、可选的零权重极性一致项和其他结构保持原样。

## 兼容性和测试

清单显式记录 `classification_variant`；读取历史清单时缺省为 `flat`。保存权重复算 valid 必须按清单重建对应分类头并继续 `strict=True` 加载。配置拒绝未知变体。CORN 单元测试必须验证：概率有限、和为一、argmax 有效；两条条件 BCE 只在正确的样本集合上计算；不可用原始模态值仍不影响输出；假归档的 `test` 哨兵在 `run_q2` 和保存权重 valid 重建路径中始终不可访问。

## 一次性运行与判定

实现、测试和两阶段审查完成后，使用新的忽略配置 `q2-corn.toml` 先运行 `train-q2 --check`，要求 `3395/728/30`，再仅训练一次到 `/home/administrator/MyItem/E/artifacts/q2-valid-corn/`。输出需含非空模型和清单、728 条 valid 报告、27 个缺失场景和 30 条附件 3 预测。比较清单时把 A 的历史缺省字段规范化为 `polarity_consistency_loss_weight=0.0`、`fusion_variant="gated"`、`text_adapter_variant="identity"` 和 `classification_variant="flat"`，唯一差异必须为 `classification_variant`。

接受条件只有 clean valid macro-F1 `>=0.6212527658`。不论结果，记录四项 clean 指标、逐类 F1、场景 mean/worst 和完整输入契约；未达标则永久淘汰该精确 CORN 分类头，不调条件损失缩放、类别权重、阈值、回归权重或 checkpoint。
