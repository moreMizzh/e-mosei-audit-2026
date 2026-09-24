# 问题 2：valid 两点提升优化设计

## 目标

在固定 `seed=20260924` 的单次实验中，找到 clean valid macro-F1 至少为 `0.6212527658` 的候选，即相对有效 v4 基线 `0.6012527658` 提升至少 `0.02`。不进行多随机种子平均。

## 固定边界

- 模型选择只使用附件 2 `train=3395` 和 `valid=728`。运行不会索引或验证 `test` 键；同一 pickle 根映射的反序列化不可避免，但 test 标签、预测、指标、早停和候选选择均不使用。
- 原始归档、附件 2/3 和既有 `q2-default*` 产物只读。每个候选写入新的、此前不存在的 `artifacts/` 目录。
- 固定本地 BERT、对齐版 `aligned_50.pkl`、连续局部缺失策略、30 epochs、batch size 64、learning rate 0.001、weight decay 0.0001、hidden size 128、4 heads、2 layers、CUDA 和 seed。
- 每次训练前执行 `train-q2 --check`；不重新运行附件 2 test，不以附件 3 无标签预测作模型选择。

## 已有证据

- v4 的 clean valid macro-F1 为 `0.6012527658`，MAE 为 `0.6246957183`。Neutral F1 `0.4817518` 是三类中最低。
- dropout `0.2` 的候选 macro-F1 为 `0.6037977`，仅提升 `0.0025450`，不满足新门槛；Neutral F1 降至 `0.4654731`，因此不继续沿 dropout 调参。
- 分类优先损失候选完成：`regression_loss_weight=0.25` 的 clean valid macro-F1 为 `0.6010339501`，较 v4 下降 `0.0002188157`，未达到 `0.6212527658`。准确率提高 `0.0109890110`，但 Neutral F1 从 `0.4817518` 降至 `0.4383562`，MAE 增加 `0.0235345960`；因此淘汰该候选，不重复或扩展该方向。审计确认 `728` 个 valid 样本、`27` 个场景行、`30` 条附件 3 预测，且输入引用、seed 和归一化统计与 v4 一致。比较记录：`artifacts/q2-valid-comparison-v4-regression-loss-025.json`。
- 容量候选完成：`hidden_size=256` 的 clean valid macro-F1 为 `0.5883282663`，较 v4 下降 `0.0129244995`；准确率下降 `0.0109890110`，MAE 增加 `0.0134999752`，Pearson 下降 `0.0263313645`。Neutral F1 `0.4782609` 仍低于 v4，故淘汰容量方向。审计同样确认 `728` 个 valid 样本、`27` 个场景行、`30` 条附件 3 预测，输入引用、seed 和归一化统计与 v4 一致。比较记录：`artifacts/q2-valid-comparison-v4-hidden-256.json`。
- 类别权重指数候选完成：`class_weight_exponent=1.25` 的 clean valid macro-F1 为 `0.5910328778`，较 v4 下降 `0.0102198880`；Neutral F1 由 `0.4817518` 降至 `0.4294118`，故淘汰该方向。审计确认 `728` 个 valid 样本、`27` 个场景行、`30` 条附件 3 预测，输入引用、seed 和归一化统计与 v4 一致。比较记录：`artifacts/q2-valid-comparison-v4-class-weight-125.json`。
- v4 保存权重的只读 valid-logit 诊断表明：在 `[-2.0, 2.0]`、步长 `0.01` 的偏置网格中，Neutral 单偏置的最高 macro-F1 为 `0.6050859145`；同时调整 Neutral/Positive 的最高值为 `0.6099789379`，仍低于门槛。因此不把后处理决策校准作为后续候选。记录：`artifacts/q2-valid-v4-decision-bias-diagnosis.json`。

## 候选顺序

每次只改变一个可解释变量，记录所有四个 clean valid 指标及三类报告。候选达到门槛后立即停止。

1. **分类优先损失平衡（已淘汰）**：新增显式 `regression_loss_weight`，从现有隐式值 `0.5` 调为 `0.25`，输出 `artifacts/q2-valid-regression-loss-025`。保持分类交叉熵、类别权重、架构和 dropout `0.1` 不变。其 macro-F1 下降，停止该方向。
2. **容量（已淘汰）**：将 `hidden_size` 从 `128` 改为 `256`，输出 `artifacts/q2-valid-hidden-256`。其他参数回到 v4，包括 `regression_loss_weight=0.5`。其四个 clean valid 指标均未改善，停止该方向。
3. **类别权重指数（已淘汰）**：令类别权重为 `w_c(alpha) = n_c^(-alpha) * N / sum_j n_j^(1-alpha)`，令 `alpha=1.25`，输出 `artifacts/q2-valid-class-weight-125`。`alpha=1.0` 与原有 `N / (3*n_c)` 公式严格相同，归一化使每个训练样本的平均权重保持为 1。其 Neutral F1 与 macro-F1 都下降，停止该方向。
4. **训练合成缺失（下一候选）**：关闭训练阶段随机连续模态缺失，仅保留原始可用性掩码；输出 `artifacts/q2-valid-no-train-missingness`。其余训练参数完全回到 v4，并保持验证场景和附件 3 推理的审计输出。理由是前三个参数方向均未改善 clean valid，而原始训练每个 batch 都人为遮蔽 1 至 2 个模态的 `10%` 至 `50%` 可用段，可能与当前 clean-only 目标冲突。

## 第四候选结果

- `synthetic_missingness_enabled=false` 的 clean valid macro-F1 为 `0.6165677806`，较 v4 提升 `0.0153150148`；accuracy 提升 `0.0274725275`，MAE 改善 `0.0033552051`。它尚差门槛 `0.0046849852`，故不宣称完成两点提升。
- 27 个连续缺失场景的 mean macro-F1 从 `0.5889129309` 提升至 `0.6037955229`，最差场景从 `0.5409674554` 提升至 `0.5463436545`，未出现 clean/鲁棒性取舍。该候选可作为后续单变量处理的顺序对照。
- 完整性审计为 valid `728` 条、场景 `27` 行、附件 3 预测 `30` 条，模型和清单均存在；归档、7-Zip、BERT、seed 与归一化统计和 v4 相同。比较记录：`artifacts/q2-valid-comparison-v4-no-train-missingness.json`。

## 实现与验证

- `Q2Config` 增加必须显式指定的 `regression_loss_weight`，默认示例配置写 `0.5` 以保持基线语义；配置要求其为有限非负数。
- 训练损失改为 `classification_loss + regression_loss_weight * smooth_l1_loss`；运行清单记录该参数。用纯单元测试锁住默认权重、非负校验和损失系数的实际乘法。
- 类别权重指数必须显式配置，要求有限且大于零；默认 `1.0` 保持既有权重。清单记录该指数，并用单元测试锁住 `alpha=1.0` 的既有权重与非默认指数下的归一化。
- 每个候选的清单必须与 v4 比较，除当前实验变量、最佳 epoch 和由其产生的模型状态/指标外，训练参数与输入引用相同。
- 验证候选有 728 条 valid 分类报告、27 行缺失场景、30 条附件 3 推理、模型和清单；全量测试必须通过。
- 单次固定 seed 结果仅是筛选证据。即使达到 valid 门槛，也不声称泛化提升或 test 提升。
