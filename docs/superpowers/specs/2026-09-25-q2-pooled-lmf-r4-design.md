# Q2 Pooled LMF-r4 Candidate Design

## 目标与边界

从当前顺序对照 A（`artifacts/q2-valid-no-train-missingness`）独立派生候选 L。唯一可配置训练差异是 `fusion_variant: "gated" -> "pooled_lmf_r4"`；在附件 2 的官方 `train=3395`、`valid=728` 划分上，固定 `seed=20260924` 执行一次训练。候选只在 clean valid `macro-F1 >= 0.6212527658` 时通过，否则永久淘汰该精确结构。

附件 2 `test` 不得被读取、索引、验证、训练、早停、选模、评估或报告。附件 3 仅生成 30 条无标签推理；它不参与任何门槛或选择。不得混合、复试或调整已经淘汰的 J（text-anchor residual）、K（pairwise Hadamard residual）、MAG-lite、MulT-lite、late-expert、CORN、adapter、学习率、损失、权重、checkpoint 或校准路线。

## 选择理由

A 是当前 clean valid 最优的顺序对照，macro-F1 为 `0.6165677806`，距离硬门槛 `0.0046849852`。J 只增加一阶 text residual，K 只增加二阶逐时槽 Hadamard residual；两者均未达到门槛。因此 L 只检验尚未覆盖的三模态三阶交互，而不叠加或微调已有残差。

低秩多模态融合先分别投影各模态、再作逐元素乘积和秩求和，避免显式构造完整三阶张量；这一参数化来自 [Low-rank Multimodal Fusion](https://aclanthology.org/P18-1209/)。此处固定 rank 为论文效率比较所用的 `4`，不对本数据进行 rank sweep。

## 精确模型变化

令现有投影后、已清除不可用位置及投影 bias 的三个状态为 `z_t, z_a, z_v`，均为 `[B, T, D]`。令 `p_m = temporal_mask & availability_m`，其中 `availability_m` 沿用既有独立掩码，不能由数值零段重新推断。对每个模态计算：

```text
l_m = sum_t p_m
h_m = sum_t (p_m * z_m) / max(l_m, 1)
q   = 1[l_t > 0 and l_a > 0 and l_v > 0]
```

L 仅新增三组无 bias、固定秩 `R=4` 的因子 `W_m[r] in R^(D x D)`：

```text
u_m[r] = h_m @ W_m[r]
r_L    = q * sum_{r=1..4}(u_t[r] * u_a[r] * u_v[r])
g_L    = g_A + r_L
```

`g_A` 是现有 gated 融合经过原有时序编码器与原有注意力池化后的 `[B, D]` 向量。`r_L` 只在拼接原有 availability fraction、进入原有分类/回归头之前相加。它不改变投影、gate、时序编码器、pool attention、head、loss、optimizer、scheduler、epoch、checkpoint 规则或 BERT 调用。

`R=4` 是 `pooled_lmf_r4` 名称的一部分，而不是 TOML 字段或可搜索超参数。新增参数数严格为 `3 * R * D * D`，在当前 `D=128` 时为 `196608`。因子只用逐矩阵 Xavier-uniform 初始化，且在已有 A 参数创建之后初始化，保证相同 seed 下 A 的既有参数位完全一致。运行清单在 `architecture.pooled_lmf_rank` 中记录 `4`，训练配置仍只有 `fusion_variant` 一项差异。

## 不变量与失败处理

- 任一模态无有效时槽时 `q=0`；L 的 logits、score、gates 与 temporal attention 必须逐元素等于同权重 A。
- 改动任一不可用原始 text/audio/vision 值、未掩码的 projection 输出或 padding 值都不能改变 L 的输出。
- append padding 不得改变 `h_m`、`r_L` 或预测。零段只保留为数值证据，不能等同于不可用。
- `q=0` 样本的所有新增因子梯度必须为零；三模态均可用时才允许新增路径影响预测。
- L checkpoint 必须按 `fusion_variant="pooled_lmf_r4"` 使用 `strict=True` 重建；历史 manifest 缺少此字段时继续默认 A 的 gated 架构。
- 若预检数量不是 `3395/728/30`、输出目录已存在、任一有效性测试失败、清单/归一化器不一致或候选已被运行，停止且不启动真实训练。

## 评估和留档

完成全套测试、独立的 fake-archive valid-only sentinel、`train-q2 --check` 后，执行一次非 `--check` 训练。审计产物必须包括模型、运行清单、728 条 valid 分类报告、四项 clean valid 指标、27 条缺失场景和 30 条附件 3 预测。比较 JSON 必须只以 A 为基准，明确记录三阶因子 rank、参数数、训练清单差异、normalizer 一致性、数量核验、每类 F1、场景均值/最差值以及不使用 test 的范围。

若未达到 `0.6212527658`，README 的 Val 表记录真实四项指标，Test 表同一模型行全部为 `未评估`；探索组合表记录拒绝理由，且不调整 rank、初始化、weight decay、loss、学习率、checkpoint 或残差缩放。
