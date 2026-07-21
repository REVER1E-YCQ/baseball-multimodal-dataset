"""使用训练完成的 BEATs 模型预测单个音频的 fly_ball / ground_ball。

示例：
    python predict_beats_classifier.py --model outputs/beats_demo/best_model.pt \
      --audio ../dataset/fly_ball/Codex_Workstation/F_001/audio.wav --impact-time 0.99

这个分类器当前需要已知或由定位器给出的击球时间。若直接把 6 秒完整音频送入模型，
背景解说/观众声会明显削弱实验结论。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.beats_classifier import BEATsBinaryClassifier
from train_beats_classifier import read_centered_clip


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="预测一段已对齐击球音频的类别")
    parser.add_argument("--model", required=True, help="训练输出的 best_model.pt")
    parser.add_argument("--audio", required=True, help="待预测的 audio.wav")
    parser.add_argument("--impact-time", required=True, type=float, help="击球时间（秒）；可用 event_start/end 的中点")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.model, map_location=device, weights_only=False)
    config = saved["config"]
    # 新检查点会明确记录验证集选出的池化方式；旧检查点没有该字段时保持 mean，
    # 从而兼容此前只使用时间平均池化训练的模型。
    pooling_mode = saved.get("pooling_mode", config.get("model", {}).get("pooling_mode", "mean"))
    model = BEATsBinaryClassifier(
        PROJECT_ROOT / config["paths"]["beats_checkpoint"],
        head_dropout=float(config["model"]["head_dropout"]),
        unfreeze_last_blocks=int(config["model"]["unfreeze_last_blocks"]),
        pooling_mode=pooling_mode,
    ).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    clip = read_centered_clip(
        Path(args.audio),
        args.impact_time,
        float(config["audio"]["crop_seconds"]),
        int(config["audio"]["target_sample_rate"]),
        remove_dc_offset=bool(config["audio"]["remove_dc_offset"]),
        peak_normalize=bool(config["audio"]["peak_normalize"]),
    ).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(clip).softmax(dim=1)[0].cpu().tolist()
    result = {"fly_ball": probabilities[0], "ground_ball": probabilities[1]}
    result["prediction"] = max(result, key=result.get)
    result["pooling_mode"] = pooling_mode
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
