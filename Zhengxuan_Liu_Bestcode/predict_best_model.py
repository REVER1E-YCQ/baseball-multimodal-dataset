from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import librosa
import numpy as np

from train_best_model import Config, ID_TO_LABEL, extract_event_clip, traditional_features


def detect_event_center(audio: np.ndarray, sample_rate: int) -> float:
    frame_length = max(32, int(round(0.025 * sample_rate)))
    hop_length = max(16, int(round(0.005 * sample_rate)))
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    ).reshape(-1)
    peak_frame = int(np.argmax(rms))
    return (peak_frame * hop_length + frame_length / 2) / sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one baseball-hit WAV file.")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("best_model.joblib"))
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--event-center", type=float)
    parser.add_argument("--event-start", type=float)
    parser.add_argument("--event-end", type=float)
    args = parser.parse_args()

    if (args.event_start is None) != (args.event_end is None):
        parser.error("--event-start and --event-end must be provided together")
    if args.event_center is not None and args.event_start is not None:
        parser.error("Use either --event-center or --event-start/--event-end")

    cfg = Config()
    audio, sample_rate = librosa.load(args.audio, sr=None, mono=True)
    if sample_rate != cfg.sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=cfg.sample_rate)
        sample_rate = cfg.sample_rate

    if args.event_center is not None:
        event_center = args.event_center
        event_source = "provided_center"
    elif args.event_start is not None:
        event_center = (args.event_start + args.event_end) / 2.0
        event_source = "provided_interval"
    else:
        event_center = detect_event_center(audio, sample_rate)
        event_source = "automatic_rms_peak"

    clip = extract_event_clip(audio, sample_rate, event_center, cfg)
    feature_vector = traditional_features(clip, cfg).reshape(1, -1)
    model = joblib.load(args.model)
    predicted_id = int(model.predict(feature_vector)[0])
    probabilities = model.predict_proba(feature_vector)[0]
    classes = model.named_steps["svc"].classes_.astype(int)
    probability_by_label = {
        ID_TO_LABEL[int(class_id)]: float(probability)
        for class_id, probability in zip(classes, probabilities)
    }
    result = {
        "audio": str(args.audio.resolve()),
        "event_center_seconds": float(event_center),
        "event_source": event_source,
        "predicted_label": ID_TO_LABEL[predicted_id],
        "probabilities": probability_by_label,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
