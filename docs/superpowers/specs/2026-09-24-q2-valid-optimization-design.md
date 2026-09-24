# 问题 2：valid-only 单变量优化实验设计

## 目标

在不再查看或使用附件 2 test 指标的条件下，以固定随机种子进行顺序的单变量实验，直到找到一个 clean valid macro-F1 严格超过 v4 基线的候选。为节省计算，不进行多随机种子平均。

## 固定边界

- 原始分卷归档、附件 2 和附件 3 均只读；既有 `q2-default*` 产物不可覆盖。
- 本轮模型选择只使用附件 2 `train=3395` 与 `valid=728`。运行时不会索引或验证 `test` 键；由于三个划分封装在同一 pickle，反序列化根映射是必要的输入读取，但 test 标签、预测和指标均不参与训练、早停、候选选择或报告。
- 附件 2 test 的后验指标冻结在 `artifacts/q2-test-evaluation-v4.json`，本实验不重新运行 test。
- 保持对齐版 `aligned_50.pkl`、本地冻结 `bert-base-uncased` 和连续局部缺失训练策略。固定 `seed=20260924`、30 epochs、batch size 64、learning rate 0.001、weight decay 0.0001、hidden size 128、4 heads、2 layers、CUDA。
- 输出目录必须是此前不存在的新目录，运行前通过 `train-q2 --check`。

## 基线和误差画像

有效基线是 `artifacts/q2-default-v4`，其 clean valid 指标为：Accuracy `0.6112637363`、macro-F1 `0.6012527658`、MAE `0.6246957183`、Pearson `0.6179443855`。旧的 `q2-default` 与 `q2-default-v2` 存在已知掩码泄漏，不能用于本轮比较。

从保存的 v4 权重生成 `artifacts/q2-valid-evaluation-v4.json`，记录 728 条 valid 样本的混淆矩阵和逐类 precision、recall、F1。该产物可由 `classification_report` 复核，且不含 test 指标。

## 顺序实验

每次只改变一个可解释变量，所有候选与 v4 以同一固定 seed 的 clean valid 结果比较：

1. 唯一候选仅将 dropout 从 `0.1` 调整为 `0.2`，输出至 `artifacts/q2-valid-dropout-020`。
2. 候选 macro-F1 严格大于 `0.6012527658` 后，立即停止本轮探索，不再训练第二个候选，也不使用 test 决策。

MAE、Accuracy 和 Pearson 与 macro-F1 一并记录。若 macro-F1 的微小提升伴随 MAE 明显恶化，候选只可称为“分类指标突破”，不能称为两个任务均优，也不自动替代 v4 作为回归基线。

## 验证

- `classification_report` 的纯单元测试必须检查固定标签/预测的三类混淆矩阵和每类 precision、recall、F1，并拒绝 0/1/2 之外的类别。
- 每个成功候选写出 `metrics.json`、`valid_classification_report.json`、27 行 `validation_scenarios.csv`、30 行附件 3 推理、模型和运行清单。
- 运行后核对候选清单与 v4 清单：除 `dropout` 和由训练产生的最佳 epoch 外，固定训练参数和输入引用必须一致。
- 完成后重新运行默认测试；报告只给出 valid 比较与单次运行的随机性限制，不重新运行 test。
