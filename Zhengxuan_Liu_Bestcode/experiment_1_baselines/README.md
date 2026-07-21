# 实验一：传统特征与两种 CNN 基线

本实验在 2135 条棒球击球声音频上比较三条基础路线：

1. 683 维传统人工特征 + StandardScaler + 类别平衡 RBF-SVM
2. log-Mel 频谱 + 二维 CNN
3. 原始音频波形 + 一维 CNN

固定划分为 1494 条训练数据和 641 条测试数据。三种方法中，传统 RBF-SVM 表现最好，测试 Macro-F1 为 `0.6129`。

## 文件说明

- `run_three_audio_methods.py`：三种方法的完整训练与评估程序
- `FULL_EXPERIMENT_REPORT_zh.md`：三种方法的完整中文实验报告
- `metrics.json`：三种方法的测试指标
- `method_comparison.csv`：方法对比表
- `train_best_model.py`：重新训练最佳传统模型
- `predict_best_model.py`：使用传统模型预测单条音频
- `best_model.joblib`：训练完成的传统 RBF-SVM
- `MODEL_RESULTS.md`：最佳传统模型的独立说明

## 运行三种方法

在仓库根目录执行：

```bash
python Zhengxuan_Liu_Bestcode/experiment_1_baselines/run_three_audio_methods.py \
  --dataset-root dataset \
  --output-dir outputs/experiment_1_baselines \
  --cache-dir outputs/experiment_1_cache
```

第二次 EAT 实验会复用该命令生成的 `dataset_split.csv` 和 `prepared_arrays.npz`，以确保两次实验的数据与传统特征完全一致。
