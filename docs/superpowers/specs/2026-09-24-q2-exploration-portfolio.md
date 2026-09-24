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

## 阶段记录

- 候选 A 已完成：关闭训练合成缺失得到 clean macro-F1 `0.6165677806`，较 v4 提升 `0.0153150148`，且 27 场景 mean/worst macro-F1 同时提升。它未达到 `0.6212527658`，但成为 B-D 的顺序控制对照；后续每轮只在 A 上额外改变一个预注册处理。
- 候选 B 已淘汰：在 A 上加入固定 `polarity_consistency_loss_weight=0.10` 后，clean macro-F1 为 `0.6054532088`，较 A 下降 `0.0111145718`；Neutral F1 从 `0.4890109890` 降至 `0.4573170732`，MAE 从 `0.6213405132` 恶化至 `0.7194830775`。即使最差缺失场景 F1 略升，也不满足 clean/Neutral/MAE 护栏，因此不调该系数。比较记录：`artifacts/q2-valid-comparison-no-train-missingness-consistency-010.json`。
- 候选 C 已淘汰：在 A 上仅将 `fusion_variant` 从历史兼容的 `gated` 改为 `mag_lite`，clean macro-F1 为 `0.6007508180`，较 A 下降 `0.0158169626`，未达 `0.6212527658`。虽然 MAE 从 `0.6213405132` 改善至 `0.5781278610`，27 个缺失场景的 mean/worst macro-F1 从 `0.6037955229/0.5463436545` 降至 `0.5781485962/0.4818639743`。不改 MAG 的宽度、深度或缩放；进入 MulT-lite。比较记录：`artifacts/q2-valid-comparison-no-train-missingness-mag-lite.json`。
- 候选 D 已淘汰：在 A 上仅将 `fusion_variant` 改为 `mult_lite`，clean macro-F1 为 `0.5884832446`，较 A 下降 `0.0280845359`，MAE 增至 `0.6525162458`。27 个缺失场景的 mean/worst macro-F1 从 `0.6037955229/0.5463436545` 降至 `0.5779645629/0.5249651686`，均越过 A-minus-`0.01` 护栏。不改 MulT-lite 的 head、深度、dropout 或残差缩放；进入独立的冻结文本输出 adapter。比较记录：`artifacts/q2-valid-comparison-no-train-missingness-mult-lite.json`。
- 候选 F 已淘汰：从 A 独立派生，仅将 `text_adapter_variant` 从 `identity` 改为冻结 BERT 后的 `houlsby_output_b32`。clean macro-F1 为 `0.6180331930`，较 A 提升 `0.0014654124`，但仍低于 `0.6212527658` 硬门槛；MAE 为 `0.6366817355`，仍低于 v4 MAE + `0.02` 护栏。27 个缺失场景的 mean/worst macro-F1 为 `0.6041805575/0.5517189624`，较 A 分别提升 `0.0003850346/0.0053753079`。这只是正向信号，不是可接受结果；不得调 adapter 的瓶颈、激活、初始化或其他超参数，下一项进入 late-expert fusion。比较记录：`artifacts/q2-valid-comparison-no-train-missingness-text-adapter-b32.json`。
- 候选 G 已淘汰：从 A 独立派生，仅将 `fusion_variant` 从 `gated` 改为无新增参数的 `late_expert_shared`。clean macro-F1 为 `0.6026841062`，较 A 下降 `0.0138836744`，远低于 `0.6212527658` 硬门槛；MAE 改善至 `0.6153963804`，Pearson 提升至 `0.6308207051`。27 个缺失场景 mean/worst macro-F1 为 `0.5944394626/0.5758645092`，较 A 为 `-0.0093560603/+0.0295208546`，仍通过 A-minus-`0.01` 护栏，但不能替代 clean F1。不得调整 shared expert 的编码、coverage、gate、head、初始化或 checkpoint。候选 E 的教师/学生或缺失课程条件也未触发，因为 clean 与场景均值没有呈现“clean 提升、鲁棒性下降”的明确冲突。比较记录：`artifacts/q2-valid-comparison-no-train-missingness-late-expert-shared.json`。

## 候选组合

| 顺序 | 处理 | 假设与文献依据 | 工程大小 | 硬门槛 |
| --- | --- | --- | --- | --- |
| A（完成） | 关闭训练合成缺失 | 当前每个训练 batch 会遮蔽 1-2 个模态的 10%-50% 连续段；ModDrop 说明这类增强为鲁棒性服务，但完整输入性能可能与鲁棒性存在权衡。该对照的 clean 与场景 F1 均提高。[[ModDrop]](https://doi.org/10.1109/TPAMI.2015.2461544) | 小 | 未达两点门槛，但作为后续顺序对照。 |
| B（淘汰） | 极性-强度一致性损失 | 让连续回归头与三分类期望极性一致，直接利用现有标签与双头，针对 Neutral 边界而不加模型参数。多任务情感分解可行性见 [[Tian et al.]](https://aclanthology.org/W18-3306/)。固定 `0.10` 使 macro-F1 和 MAE 同时恶化。 | 小 | 不调系数，进入 MAG-lite。 |
| C（淘汰） | MAG-lite 非语言残差 | 先用掩码感知的 audio/vision 残差更新投影后的 text，再保留现有门控和时序 Transformer。MAG 的机制是向预训练语言表示注入非语言条件偏移；本项目冻结 BERT，因而只验证简化后置版本。该固定结构使 clean 与缺失场景 F1 都回退。[[MAG]](https://aclanthology.org/2020.acl-main.214/) | 中 | 单层固定结构，不解冻 BERT；不做后续 MAG 调参。 |
| D（淘汰） | MulT-lite 定向交叉注意力 | 在当前“先凸门控、后时序编码”的信息压缩前保留模态身份，以 text<-audio、text<-vision 定向注意力建模跨时交互。每方向一层且全 K/V 缺失时零更新；该固定结构使 clean F1、MAE 与缺失场景均回退。[[MulT]](https://aclanthology.org/P19-1656/) | 中高 | 不调 head、深度、dropout 或残差缩放。 |
| E（未触发） | 受控缺失课程或完整教师-遮蔽学生蒸馏 | 仅在 G 后显示 clean/鲁棒明确冲突时使用。G 的 clean 与场景均值均下降，故本候选不作为对 G 的调参替代。[[UMDF]](https://ojs.aaai.org/index.php/AAAI/article/view/28871/) [[TFR-Net]](https://doi.org/10.1145/3474085.3475585) | 中 | 保持不执行；不能用作回调 G 的方式。 |
| F（淘汰） | 冻结 BERT 的输出残差 adapter | 在 BERT 输出、文本投影之前加入固定瓶颈 32 的残差 adapter；BERT 保持 `no_grad`。实测 clean macro-F1 `0.6180331930`，较 A 仅提高 `0.0014654124`，未达硬门槛；缺失场景 mean/worst 有小幅提升。[[Adapters]](https://proceedings.mlr.press/v97/houlsby19a.html) | 中 | 不调瓶颈、激活、初始化或其他 adapter 超参数。 |
| G（淘汰） | 时序 shared late-expert fusion | 各模态以共享时序编码得到独立 expert logits，再以 availability-masked reliability gate 融合。实测 clean macro-F1 `0.6026841062`，虽然 MAE 与最差缺失场景改善，但未达 hard gate。[[TFN]](https://aclanthology.org/D17-1115/) | 中 | 不调整编码、coverage、gate、head、初始化或 checkpoint。 |

## 不作为首轮的路线

- MISA-lite 需要 shared/private 分解、重构和多个辅助系数；本地训练集小且单 seed，不应先进入多损失搜索。[[MISA]](https://arxiv.org/abs/2005.03545)
- 监督对比损失可在 B 后作为独立候选，但 64 的 batch、人工缺失和固定单次实验会提高正样本不足与系数选择风险。[[SupCon]](https://papers.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
- Focal/effective-number 权重是合理的分类不平衡工具，但本地增大逆频率权重已损害 Neutral F1，故排在表示与一致性路线之后。[[Class-Balanced Loss]](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)
- 不在同一 valid 集上搜索多个 checkpoint、融合权重或 MAG 超参数；任何 ensemble 只有在两个各自冻结、各自达标的候选出现后，才作为单独预注册诊断。

## 执行与判定

每个方向先有单元测试、最小 fake-archive 训练测试与 `train-q2 --check`，再运行一次真实训练。只比较 clean valid macro-F1 是否达到目标；若未达到，记录四项 clean 指标、各类 F1、27 场景摘要和输入/清单差异后进入下一项。任何未达到门槛的候选都不以 accuracy、Pearson、附件 3 输出或未运行的 test 指标替代结论。
