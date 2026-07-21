"""BEATs 骨干与二分类头。

这里刻意复用 Microsoft 官方 BEATs 实现，而不是自行重写 Transformer；本文件只负责：
1. 加载官方预训练参数；2. 冻结/有限解冻骨干；3. 对时序表示做指定池化后完成 fly/ground 二分类。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_BEATS_DIR = PROJECT_ROOT / "third_party" / "unilm" / "beats"

# 官方 BEATs.py 使用同目录导入 backbone.py，因此必须把该目录放到模块搜索路径首位。
if str(OFFICIAL_BEATS_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_BEATS_DIR))

from BEATs import BEATs, BEATsConfig  # noqa: E402  （来自官方仓库）


class BEATsBinaryClassifier(nn.Module):
    """预训练 BEATs + 小型线性分类头。

    参数
    ----
    checkpoint_path:
        Microsoft 官方发布的“预训练” BEATs checkpoint 路径。不要用 AudioSet 已分类
        checkpoint 的 527 类预测头；本任务只使用其通用声学表示。
    head_dropout:
        分类头前的 dropout 概率。数据很少时 0.2--0.5 常有助于防过拟合；过大则会
        让训练不稳定。默认 0.35。
    unfreeze_last_blocks:
        解冻 BEATs 最后 N 个 Transformer block。0 表示完全冻结，只训练约千级参数的
        分类头，是小数据实验的默认安全选择。1--2 可在验证集停滞时尝试，但会显著增加
        过拟合风险，且必须使用更低的 backbone learning rate。
    pooling_mode:
        ``mean`` 表示时间平均池化，综合整段音频的稳定信息；``max`` 表示逐特征维
        取时间最大值，更强调短促冲击，但也可能放大欢呼或解说爆音。训练脚本会在
        相同划分上分别训练两种模式，并仅使用验证集 Macro-F1 选择最终模型。
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        head_dropout: float = 0.35,
        unfreeze_last_blocks: int = 0,
        pooling_mode: str = "mean",
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"未找到 BEATs 权重：{checkpoint_path}\n"
                "请先执行 README 中的下载命令，或检查 config.json 的 paths.beats_checkpoint。"
            )

        # 权重来自 Microsoft 官方仓库；weights_only=False 是为了兼容其 checkpoint 元数据。
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(checkpoint["cfg"])
        # 即使误下载了 AudioSet 微调权重，也关闭原来的 527 类预测器，统一输出声学特征。
        cfg.finetuned_model = False
        self.backbone = BEATs(cfg)
        incompatible = self.backbone.load_state_dict(checkpoint["model"], strict=False)

        # 预训练 checkpoint 不应缺少骨干层；仅 predictor 的键可能因为上面关闭而多余。
        unexpected_non_predictor = [k for k in incompatible.unexpected_keys if not k.startswith("predictor")]
        missing_non_predictor = [k for k in incompatible.missing_keys if not k.startswith("predictor")]
        if unexpected_non_predictor or missing_non_predictor:
            raise RuntimeError(
                "BEATs checkpoint 与官方代码不匹配。\n"
                f"unexpected={unexpected_non_predictor}, missing={missing_non_predictor}"
            )

        self.embedding_dim = cfg.encoder_embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(head_dropout),
            nn.Linear(self.embedding_dim, 2),  # 0=fly_ball，1=ground_ball
        )
        self.unfreeze_last_blocks = unfreeze_last_blocks
        if pooling_mode not in {"mean", "max"}:
            raise ValueError(f"pooling_mode 只支持 mean 或 max，实际为：{pooling_mode}")
        self.pooling_mode = pooling_mode
        self._configure_trainable_parameters()

    def _configure_trainable_parameters(self) -> None:
        """冻结完整骨干，并按需仅解冻最后几个 Transformer block。"""
        if self.unfreeze_last_blocks < 0:
            raise ValueError("unfreeze_last_blocks 不能小于 0。")

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        if self.unfreeze_last_blocks:
            blocks = self.backbone.encoder.layers
            if self.unfreeze_last_blocks > len(blocks):
                raise ValueError(
                    f"BEATs 只有 {len(blocks)} 个 block，不能解冻 {self.unfreeze_last_blocks} 个。"
                )
            for block in blocks[-self.unfreeze_last_blocks :]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

        # 分类头必须始终可训练。
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True):
        """在默认“冻结骨干”模式下，仍强制骨干保持 eval 状态。

        ``nn.Module.train()`` 会递归打开 BEATs 内部 dropout。对于完全冻结的预训练骨干，
        这只会让同一条音频在每个 epoch 产生随机表示，却不会学习到更好的骨干参数。因此
        unfreeze_last_blocks=0 时关闭骨干 dropout；一旦用户解冻 block，则恢复正常训练行为。
        """
        super().train(mode)
        if mode and self.unfreeze_last_blocks == 0:
            self.backbone.eval()
        return self

    def forward(self, waveform_16khz: torch.Tensor) -> torch.Tensor:
        """输入 ``[batch, time]`` 的 16 kHz 单声道波形，输出 ``[batch, 2]`` logits。"""
        if waveform_16khz.ndim != 2:
            raise ValueError(f"期望 [batch, time]，实际得到 {tuple(waveform_16khz.shape)}")

        # 本项目每个样本都被裁剪/补零到固定时长，因此没有 padding 区域。
        padding_mask = torch.zeros_like(waveform_16khz, dtype=torch.bool)
        # 官方 BEATs 的 Kaldi fbank 前端与部分 Transformer 运算在本机 FP16 下会产生 NaN。
        # 即使外层训练启用了 autocast，这里也强制骨干以 FP32 运行；分类头仍可按外层设置
        # 使用混合精度。这样既避免静默数值崩溃，也保留以后单独优化分类头的可能。
        device_type = waveform_16khz.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            features, feature_padding_mask = self.backbone.extract_features(
                waveform_16khz.float(), padding_mask=padding_mask
            )

        # 当前样本均为固定长度，没有 padding；下面仍保留变长 batch 的安全处理。
        if self.pooling_mode == "mean":
            if feature_padding_mask is not None and feature_padding_mask.any():
                valid = (~feature_padding_mask).unsqueeze(-1).to(features.dtype)
                pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            else:
                pooled = features.mean(dim=1)
        else:
            if feature_padding_mask is not None and feature_padding_mask.any():
                # 最大池化前把 padding 帧设为极小值，避免补零区域被错误选为最大响应。
                masked = features.masked_fill(
                    feature_padding_mask.unsqueeze(-1), torch.finfo(features.dtype).min
                )
                pooled = masked.amax(dim=1)
            else:
                pooled = features.amax(dim=1)
        return self.classifier(pooled)

    def parameter_groups(self, head_lr: float, backbone_lr: float, weight_decay: float):
        """返回不同学习率的参数组；被冻结参数不会交给优化器。"""
        head_params = [p for p in self.classifier.parameters() if p.requires_grad]
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": head_lr, "weight_decay": weight_decay}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
        return groups
