# Zhengxuan Liu Best Model Code

这个目录保存本次棒球击球声音二分类实验中综合表现最好的算法：

```text
0.5秒击球音频
-> STFT、log-Mel、MFCC和频谱/时域统计特征
-> 683维特征向量
-> StandardScaler
-> class-balanced RBF-SVM
-> ground_ball / fly_ball
```

详细测试结果见[MODEL_RESULTS.md](MODEL_RESULTS.md)。仓库内附带的`best_model.joblib`是使用1494条训练数据得到的最终模型。

## 文件说明

- `train_best_model.py`：从`dataset/`重新划分数据、提取特征、搜索参数并训练模型。
- `predict_best_model.py`：使用训练好的模型预测一条WAV音频。
- `best_model.joblib`：本次实验训练好的模型。
- `model_metadata.json`：模型参数、数据来源和测试结果。
- `requirements.txt`：Python依赖。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 重新训练

在仓库根目录运行：

```bash
python Zhengxuan_Liu_Bestcode/train_best_model.py \
  --dataset-root dataset \
  --output-dir Zhengxuan_Liu_Bestcode/retrained_output
```

训练脚本使用固定随机种子42，保持70%训练、30%测试，并确保完全相同的音频不会跨越训练集和测试集。

## 预测一条音频

如果已经知道击球事件的开始和结束时间：

```bash
python Zhengxuan_Liu_Bestcode/predict_best_model.py \
  --audio path/to/audio.wav \
  --event-start 1.20 \
  --event-end 1.28
```

如果只知道击球中心时刻：

```bash
python Zhengxuan_Liu_Bestcode/predict_best_model.py \
  --audio path/to/audio.wav \
  --event-center 1.24
```

不提供击球时间时，程序会使用短时RMS峰值自动寻找最强瞬态：

```bash
python Zhengxuan_Liu_Bestcode/predict_best_model.py --audio path/to/audio.wav
```

已知人工标注时间通常比自动寻找峰值更加可靠。
