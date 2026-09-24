# 问题 2：valid-only 多路线探索设计

## 目标

在附件 2 的 `train=3395`、`valid=728` 上，以固定 `seed=20260924` 寻找 clean valid macro-F1 至少 `0.6212527658` 的候选。当前 v4 基线为 `0.6012527658`。这是单次筛选证据，不代表测试集或泛化收益。

## 不变边界

- 训练、早停、候选选择和指标只访问 Attachment 2 的 `train` 与 `valid`；不索引、验证、预测或汇报 Attachment 2 `test`。
- 每个候选只写入一个不存在的新 `artifacts/` 目录，原始归档和既有 `q2-default*`、已淘汰候选均只读。
- 每轮只引入一项预注册处理；固定 seed、30 epochs、batch size 64、learning rate `0.001`、weight decay `0.0001`、对齐的 `aligned_50.pkl`、本地 BERT、CUDA 和现有 clean-valid checkpoint 规则。
- 每轮都保存 `run_manifest.json`、728 条分类报告、27 条缺失场景、30 条附件 3 预测和对 v4 的比较 JSON；附件 3 仅作无标签推理审计。

## 已排除方向

- `dropout=0.2` 只将 macro-F1 提至 `0.6037977`，不满足两点门槛。
- `regression_loss_weight=0.25`、`hidden_size=256` 和 `class_weight_exponent=1.25` 分别得到 `0.6010339501`、`0.5883282663`、`0.5910328778`，均淘汰。
- v4 保存权重的 valid-logit 偏置网格最高为 `0.6099789379`，不足门槛；不以 valid 后处理阈值制造表面提升。
- 不继续加大逆频率权重，不首先全量微调 BERT，也不在单 seed valid 上网格搜索损失系数或架构宽度。

## 候选组合

| 顺序 | 处理 | 假设与文献依据 | 工程大小 | 硬门槛 |
| --- | --- | --- | --- | --- |
| A | 关闭训练合成缺失 | 当前每个训练 batch 会遮蔽 1-2 个模态的 10%-50% 连续段；ModDrop 说明这类增强为鲁棒性服务，但完整输入性能可能与鲁棒性存在权衡。先建立 clean-control，再看 27 场景。[[ModDrop]](https://doi.org/10.1109/TPAMI.2015.2461544) | 小 | clean macro-F1 达目标；同时报告 27 场景均值和最差值，不把场景指标作为 checkpoint 选择。 |
| B | 极性-强度一致性损失 | 让连续回归头与三分类期望极性一致，直接利用现有标签与双头，针对 Neutral 边界而不加模型参数。多任务情感分解可行性见 [[Tian et al.]](https://aclanthology.org/W18-3306/)。 | 小 | 固定 `consistency_loss_weight=0.10`；macro-F1 达目标，Neutral F1 不低于 v4，MAE 不高于 v4 `+0.01`。 |
| C | MAG-lite 非语言残差 | 先用掩码感知的 audio/vision 残差更新投影后的 text，再保留现有门控和时序 Transformer。MAG 的机制是向预训练语言表示注入非语言条件偏移；本项目冻结 BERT，因而只验证简化后置版本。[[MAG]](https://aclanthology.org/2020.acl-main.214/) | 中 | 单层固定结构，不解冻 BERT；macro-F1 达目标且 MAE 不高于 v4 `+0.02`。 |
| D | MulT-lite 定向交叉注意力 | 在当前“先凸门控、后时序编码”的信息压缩前保留模态身份，以 text<-audio、text<-vision 等定向注意力建模跨时交互。[[MulT]](https://aclanthology.org/P19-1656/) | 中高 | 每个方向一层，所有 K/V 缺失时零更新；macro-F1 达目标，掩码测试与 27 场景完整。 |
| E | 受控缺失课程或完整教师-遮蔽学生蒸馏 | 仅在 A 显示 clean/鲁棒明确冲突时使用。课程匹配现有 10/30/50% 场景；蒸馏比重建原始特征小，符合不完整模态自蒸馏思路。[[UMDF]](https://ojs.aaai.org/index.php/AAAI/article/view/28871/) [[TFR-Net]](https://doi.org/10.1145/3474085.3475585) | 中 | 先固定一个课程或一个蒸馏系数，不网格；clean 与最差场景 F1 都必须改善。 |

## 不作为首轮的路线

- MISA-lite 需要 shared/private 分解、重构和多个辅助系数；本地训练集小且单 seed，不应先进入多损失搜索。[[MISA]](https://arxiv.org/abs/2005.03545)
- 监督对比损失可在 B 后作为独立候选，但 64 的 batch、人工缺失和固定单次实验会提高正样本不足与系数选择风险。[[SupCon]](https://papers.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
- Focal/effective-number 权重是合理的分类不平衡工具，但本地增大逆频率权重已损害 Neutral F1，故排在表示与一致性路线之后。[[Class-Balanced Loss]](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)

## 执行与判定

每个方向先有单元测试、最小 fake-archive 训练测试与 `train-q2 --check`，再运行一次真实训练。只比较 clean valid macro-F1 是否达到目标；若未达到，记录四项 clean 指标、各类 F1、27 场景摘要和输入/清单差异后进入下一项。任何未达到门槛的候选都不以 accuracy、Pearson、附件 3 输出或未运行的 test 指标替代结论。
