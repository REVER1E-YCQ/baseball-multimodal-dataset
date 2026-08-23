# M2D 精度提升尝试：探针家族与配对对比（2026-08-09/10）

**状态：** 已完成，方向关闭（无稳定增益）；**数据：** 同锁定基准快照

## 背景

headline 0.667（attention 池化 + L2 逻辑回归）建立后，检验分类头侧是否还有可挖的稳定增益。包含三个子实验，全部带严格 Pre / 瞬态移除负控和置换检验。

## 结果

**1. 探针复现与基准缝**：logistic 复现 0.667（AUC 0.693），确认 headline 可复现。

**2. 校准边际分类器**：

| 分类头 | Event BA | AUC |
|---|---:|---:|
| attention + logistic（基准） | **0.667** | 0.693 |
| attention + linear SVM（校准） | 0.657 | 0.705 |
| attention + RBF SVM（校准） | 0.652–0.659 | 0.714 |

SVM 的 AUC 更高（排序质量更好）但 BA 不超过 logistic，校准阈值也补不回来。

**3. 配对对比特征**（event vs event±delta，3 折种子）：event_alone 0.644–0.667，event_plus_delta 0.648–0.667——加入窗口间差分特征无一致增益。

## 结论

在当前特征之上换分类头或加配对差分特征，都拿不到超过 0.667 的稳定增益。**分类头方向关闭**；0.667 保持为 headline。剩余瓶颈更可能在任务模糊度（标签上限）而非模型/分类头。

**产物：** 本地 `outputs/m2d_exploratory_probes/`、`m2d_margin_classifier_evaluation/`、`m2d_paired_contrast_evaluation/`
