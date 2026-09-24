# Q2 Sinusoidal Temporal Position Candidate Design

## 目标与范围

从当前顺序对照 A（`artifacts/q2-valid-no-train-missingness`）独立派生候选 M。唯一配置差异为 `temporal_position_variant: "none" -> "sinusoidal"`；`fusion_variant` 保持 `"gated"`，以固定 `seed=20260924`、附件 2 官方 `train=3395` / `valid=728` 划分执行一次训练。clean valid `macro-F1 >= 0.6212527658` 是唯一接受门槛。

附件 2 `test` 不得被读取、索引、验证、训练、早停、选模、评估或报告。附件 3 仅保留 30 条无标签推理，不能参与选择。禁止叠加、调参或重试 J/K/L residual、MAG-lite、MulT-lite、late-expert、CORN、b32 adapter、类别权重、损失权重、学习率、scheduler、checkpoint 或校准。

## 选择依据

A 的时序 Transformer 接收门控后的 `fused[B,T,H]`，但进入编码器前没有显式位置编码；scalar temporal pool 也只按内容打分。因而有效槽序列缺少明确的顺序信号。L 的三阶低秩 residual clean macro-F1 为 `0.6003205161`，较 A 下降 `0.0162472645`；结合 J/K 的一阶/二阶 residual 失败，不再向 A 叠加交互残差。

M 只恢复 Transformer 的固定正弦位置公式，不引入可学习位置表、长度、幅度、dropout 或数据驱动选择。该定义来自 [Attention Is All You Need](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need.pdf)，底数 `10000` 是公式的一部分，不是本地搜索超参数。

## 精确变化

令 A 已有 masked gated sequence 为 `f[B,T,H]`，且 `temporal[B,T]` 表示非 padding 且至少有一条模态实际可用的位置。仅当 `temporal_position_variant == "sinusoidal"` 时，在原有 TransformerEncoder 之前计算：

```text
p[t, 2i]   = sin(t / 10000^(2i/H))
p[t, 2i+1] = cos(t / 10000^(2i/H))
f_M[b,t,:] = temporal[b,t] * (f[b,t,:] + p[t,:])
```

位置从 `t=0` 开始；奇数 `H` 时最后一个未配对维度只使用 sine 分量。编码器、`src_key_padding_mask=~temporal`、temporal attention、availability fraction、分类头、回归头、BERT、损失、optimizer、训练 epoch、checkpoint 选择规则全部保持 A 的既有实现。

该变体新增零个参数，state-dict keys 和相同 seed 下的所有 A 初始参数完全一致。新的持久字段 `temporal_position_variant` 只用于明确保存语义；历史 run manifest 缺少该字段时默认 `"none"`，并继续按 `strict=True` 重建。

## 安全不变量

- 原始 text/audio/vision 的不可用位置即使改为大有限值，也不能改变预测；位置编码只能在已有 `temporal` 槽上相加。
- 任意 padding 槽在加位置编码后必须重新清零，不能影响任一有效槽、attention、logits 或 score；append padding 保留所有原有效槽的 position index。
- 位置 helper 的值必须只由位置、hidden size、dtype 和 device 决定；无随机数、无可训练参数、无长度表或数据统计量。
- `temporal_position_variant="none"` 必须逐元素保留 A 的已有前向行为；`"sinusoidal"` 仅允许在有效位置改变进入 encoder 的时序表示。
- manifest 中的变体必须驱动 saved-valid 的严格构造；valid-only runner 的 test-split sentinel 必须继续使任何 test 读取失败。

## 测试、运行与判定

在实现前先编写 helper 数值、none 等价、编码器输入差异、padding/不可用值不变性、零新增参数/RNG、config、strict reconstruction 与 fake train/valid sentinel 测试。全套测试、`train-q2 --check` 返回精确 `3395/728/30`、且候选输出路径不存在后，才执行一次非 `--check` 训练。

真实结果必须保留模型、manifest、728 条 valid 分类报告、27 条受控缺失场景、30 条附件 3 预测及 A/M 比较 JSON。若未达到门槛，在 README Val 表写入实际四项指标、Test 表写 `未评估`，并永久禁止对位置公式、base、幅度、可学习位置、dropout、optimizer、loss、checkpoint 或 seed 进行调节。
