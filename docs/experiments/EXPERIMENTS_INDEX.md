# 音频实验总索引与思路梳理（Haoran Yan，截至 2026-08-13）

**一句话结论：** 棒球击球接触音频确实携带可复现的 fly/ground 相关信息（人工核验快照上 event BA 0.667、AUC ≈0.70、严格 Pre 负控随机、接触特异增量 +0.17）；但该信号强度中等、跨采集者迁移尚未打通，且所有表示/模型侧的改进方向已系统性探索并关闭。**当前最佳配置 = 冻结 M2D 第 11 层 attention 池化 + 校准逻辑回归，200 ms 峰值居中事件窗，来源分组 5 折。**

## 研究主线（按时间）

| 阶段 | 报告 | 核心问题 → 结论 |
|---|---|---|
| V2–V3 (07 月上旬) | `v2/v3_fly_ground_experiment_report_zh.md`（未随本分支上传） | 信号是否存在 → 弱可分，发现严重采集混淆 |
| V4 统一基准 (07-16) | `v4_unified_frozen_audio_benchmark_zh.md` | 统一协议下哪个冻结编码器最稳 → M2D；但长窗口高分主要来自上下文而非击球 |
| V5 大数据重跑 (07-19) | `v5_m2d_traditional_update_zh.md` | 2000 条 Codex 集上是否成立 → 成立（0.606），但外部人工数据迁移失败（0.490） |
| 可移植 baseline (07-31) | `model/m2d_audio_baseline/README.md` | 最小可复现包：切窗→冻结 M2D→linear probe + 负控 |
| 锁定基准 (08-05) | `20260805_short_contact_m2d_beats_benchmark_zh.md` | 人工核验快照上建立公共参照（mean 池化 0.60–0.62），负控干净 |
| 表示改进 (08-06) | `20260806_pooling_layer_threshold_zh.md` | 池化+层扫描 → **attention 第 11 层 headline 0.667**；阈值校准无收益 |
| 泄漏检查 (08-06) | `20260806_full_audio_leakage_check_zh.md` | 全音频 0.777 的来源 → 击球后内容泄漏，**禁止用全长音频报结果** |
| 对齐诊断 (08-09) | `20260809_alignment_sensitivity_zh.md` | ±50ms 掉 0.022 → 中等对齐依赖，峰值居中够用 |
| 融合 (08-09) | `20260809_encoder_fusion_zh.md` | M2D+BEATs → 负增益，弱编码器稀释强编码器 |
| 微调试点 (08-09) | `20260809_m2d_finetune_pilot_zh.md` | LoRA −0.039 → 数据规模不支持微调 |
| 分类头探索 (08-10) | `20260810_accuracy_improvement_probes_zh.md` | SVM/配对差分 → 无稳定增益，分类头方向关闭 |
| 数据增强 (08-10) | `20260810_contact_window_augmentation_zh.md` | 录音条件增强无收益，时间抖动一致变差 |
| 受控仿真探针 (08-11) | `reports/contact_synth_probe/report_zh.md` | 干净条件下物理参数可测出；广播噪声下只剩强度线索 |

## 已关闭的方向（不要重复实验）

1. **换池化/换层/换分类头/阈值校准** —— attention 第 11 层已是上限（0.667）
2. **跨编码器融合** —— 负增益
3. **LoRA 微调编码器** —— 过拟合，−0.04
4. **录音条件增强 / 时间抖动** —— 无增益或变差
5. **全长/长窗口输入** —— 结果泄漏（击球后内容），不是真信号
6. **1000 ms 以上观察窗** —— V4 已证明事件移除后分数不降

## 尚未解决（下一步候选）

- **跨采集者迁移**：V5 外部人工数据只有 87 条且同时改变类别与采集者；需要更多人工采集数据才能区分"迁移失败"与"类别不平衡"
- **任务模糊度/标签上限**：fly/ground 是粗粒度代理标签，可能本身封顶了性能
- **物理参数信息**：仿真探针表明强度类线索在噪声下最稳健，可作为特征工程方向
- 文献地图见 `docs/research/impact-audio-literature-map.md`

## 实验代码位置

所有实验的入口脚本与可复用模块都在 `model/m2d_audio_baseline/scripts/`（每个 `run_*.py` 对应一个实验，模块名见各报告末尾），配套单元测试在 `model/m2d_audio_baseline/tests/`。对应关系举例：

| 实验 | 入口 |
|---|---|
| 锁定基准 / 敏感性 | `run_m2d_primary.py`, `run_beats_primary.py`, `run_common_200ms.py`, `run_m2d_sensitivity.py` |
| 池化/层扫描/阈值 | `short_contact_benchmark.py`, `attention_control_representation.py`, `run_layer_scan.py` |
| full-audio 泄漏检查 | `run_full_audio_conditions.py` |
| 对齐敏感性 | `run_alignment_sensitivity.py` |
| 编码器融合 | `run_encoder_fusion.py` |
| LoRA 微调试点 | `run_finetune_pilot.py` |
| 分类头/配对对比 | `exploratory_probe_benchmark.py`, `margin_classifier_evaluation.py`, `paired_contrast_evaluation.py` |
| 数据增强 | `contact_window_augmentation.py` |
| 统计检验/报告 | `statistical_evidence.py`, `validate_and_report.py` |

## 防重复实验的两条铁律

1. **报结果必须带严格 Pre 负控 + 来源分组折**；Pre 接近随机才说明信号来自击球本身。
2. **任何"高得惊人"的数字先查泄漏三渠道**：随机折让同场片段进训练+测试、击球后内容编码结果、语义编码器读解说。
