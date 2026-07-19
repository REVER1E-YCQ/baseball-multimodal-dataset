from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from scripts.prepare_windows import prepare_windows


class PrepareWindowsTest(unittest.TestCase):
    def test_peak_center_and_strict_pre_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_rate = 16_000
            waveform = np.zeros(sample_rate, dtype=np.float32)
            waveform[sample_rate // 2] = 1.0
            wavfile.write(root / "audio.wav", sample_rate, waveform)
            pd.DataFrame(
                [
                    {
                        "uid": "sample_001",
                        "label": "fly_ball",
                        "source_id": "session_001",
                        "protocol_role": "primary_dev",
                        "audio_path": "audio.wav",
                        "event_start": 0.45,
                        "event_end": 0.55,
                    }
                ]
            ).to_csv(root / "manifest.csv", index=False)

            result = prepare_windows(
                root / "manifest.csv",
                root / "prepared",
                (200,),
                50,
            )

            self.assertEqual(set(result["window_name"]), {"event_200ms", "pre_200ms"})
            event = result[result["window_name"].eq("event_200ms")].iloc[0]
            strict_pre = result[result["window_name"].eq("pre_200ms")].iloc[0]
            self.assertAlmostEqual(float(event["estimated_peak_time"]), 0.5, places=6)
            self.assertAlmostEqual(float(event["window_start"]), 0.4, places=6)
            self.assertAlmostEqual(float(strict_pre["window_start"]), 0.2, places=6)
            self.assertAlmostEqual(float(strict_pre["window_end"]), 0.4, places=6)
            self.assertEqual(int(event["wav_boundary_padding_samples"]), 0)
            self.assertFalse(Path(str(event["window_path"])).is_absolute())

            event_path = (root / "prepared" / str(event["window_path"])).resolve()
            event_rate, event_waveform = wavfile.read(event_path)
            self.assertEqual(event_rate, sample_rate)
            self.assertEqual(len(event_waveform), int(0.2 * sample_rate))
            self.assertEqual(int(np.argmax(np.abs(event_waveform))), sample_rate // 10)


if __name__ == "__main__":
    unittest.main()
