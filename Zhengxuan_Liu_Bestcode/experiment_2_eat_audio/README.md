# 实验二：多尺度 EAT 与传统模型融合

本实验使用 Efficient Audio Transformer（EAT）预训练模型提取击球声音频特征。EAT 主干保持冻结，因此这里不是从零训练一个大型神经网络，而是把预训练模型当作高级音频特征提取器。

## 处理流程

```text
击球事件附近音频
-> 分别截取 0.25、0.5、1.0 秒
-> 16 kHz / 128 维 Kaldi Mel filterbank
-> 冻结的 EAT-base
-> 三尺度最大池化特征
-> 正则逻辑回归
-> 与传统 RBF-SVM 概率按 70% / 30% 融合
```

固定 641 条测试集上的最佳结果：Accuracy `0.6802`、Balanced Accuracy `0.6733`、Macro-F1 `0.6738`、ROC-AUC `0.7282`。

## 文件说明

- `run_eat_audio_fusion.py`：EAT 特征提取、候选池化与分类器选择
- `run_eat_late_fusion.py`：EAT 与传统模型的分数级融合
- `REPORT_zh.md`：完整中文实验报告
- `metrics.json`：纯 EAT 测试结果
- `late_fusion_metrics.json`：最佳融合模型测试结果
- `final_audio_method_comparison.csv`：传统、EAT 和融合方法对比
- `bootstrap_summary.json`：5000 次配对 bootstrap 结果
- `validation_model_selection.csv`：EAT 候选配置的验证结果
- `late_fusion_validation.csv`：融合权重的验证结果
- `eat_audio_fusion_model.joblib`：纯 EAT 的下游分类器
- `eat_traditional_late_fusion.joblib`：融合分类器、权重和阈值

EAT 神经网络权重没有重复提交到 Git。脚本使用固定模型版本：

```text
worstchan/EAT-base_epoch30_finetune_AS2M
revision: 60d61e8b2e9e5ba3be6860285de80cb7d625ccbb
```

## 运行方法

先运行实验一，生成完全一致的划分和传统特征缓存，然后执行：

```bash
python Zhengxuan_Liu_Bestcode/experiment_2_eat_audio/run_eat_audio_fusion.py \
  --split-csv outputs/experiment_1_baselines/dataset_split.csv \
  --traditional-cache outputs/experiment_1_cache/prepared_arrays.npz \
  --feature-cache outputs/experiment_2_eat/eat_multiscale_features.npz \
  --output-dir outputs/experiment_2_eat
```

再运行分数级融合：

```bash
python Zhengxuan_Liu_Bestcode/experiment_2_eat_audio/run_eat_late_fusion.py \
  --split-csv outputs/experiment_1_baselines/dataset_split.csv \
  --traditional-cache outputs/experiment_1_cache/prepared_arrays.npz \
  --eat-cache outputs/experiment_2_eat/eat_multiscale_features.npz \
  --eat-selection outputs/experiment_2_eat/selected_configuration.json \
  --output-dir outputs/experiment_2_eat
```

推荐使用带 CUDA 的 NVIDIA 显卡。特征提取完成后，逻辑回归和 SVM 部分也可以在 CPU 上运行。
