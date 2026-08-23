# 短接触窗口 M2D/BEATs 锁定基准（2026-08-05）

**状态：** 已锁定（ADR-0004）；**数据：** 人工核验 822 样本快照（803 条有效，fly 386 / ground 436，636 个来源谱系组）

## 问题

V4/V5 阶段（见 `v4_unified_frozen_audio_benchmark_zh.md`、`v5_m2d_traditional_update_zh.md`）在自动采集的 Codex 集上确认了弱到中等的可分信号，但标签未逐条人工核验。本实验在人工逐条核验过的快照上建立锁定基准：冻结 M2D 与 BEATs 在 200 ms 峰值居中事件窗下的对比，配严格 Pre 负控与瞬态移除负控，来源分组 5 折。

## 结果

| 配置 | Event BA | AUC | 严格 Pre | 接触特异增量 |
|---|---:|---:|---:|---:|
| M2D 40ms，mean 池化，200ms | 0.595–0.619 | 0.617–0.645 | ≈0.52–0.54 | ≈+0.09 |
| BEATs iter3+，200ms | 0.599 | 0.642 | ≈0.49（随机） | ≈+0.11 |

M2D 与 BEATs 在 mean 池化下基本打平；负控干净（Pre 与瞬态移除均接近 0.50），证明信号确实来自击球瞬态本身，而不是采集上下文。

## 结论

- 基准协议锁定：冻结编码器、200 ms 峰值居中窗、来源分组 5 折、严格 Pre + 瞬态移除双负控（ADR-0004）。
- 此基准是后续所有 8 月诊断实验（池化、层扫描、对齐、融合、微调、增强）的公共参照。

**产物：** 本地 `outputs/m2d_primary_benchmark/`、`beats_primary_benchmark/`、`common_200ms_benchmark/`、`m2d_controls_benchmark/`、`verified_snapshot_audit/`
