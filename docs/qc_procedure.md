# QC Procedure

Required gates for each accepted sample:

0. Qwen label gate: `scripts/audit_qwen_labels.py` must pass before materialization unless a human reviewer explicitly overrides the sample.
   Use `scripts/refine_qwen_events.py` before this gate to tighten coarse Qwen contact intervals with local audio peaks.
1. Schema gate: required files exist, folder label matches `label.txt`, and `sample.csv` columns match the label.
2. Media gate: `ffprobe` can read `video.mp4` and `audio.wav`; durations are close.
3. Audio-contact gate: `audio.wav` has a plausible short transient near the annotated event interval.
4. Label gate: label-specific fields are valid and confidence is above the configured threshold.
5. Source gate: `source.txt` includes `video_title:` and `video_url:`.

Recommended batch policy:

- Reject samples with missing or masked contact sound.
- Queue samples with model confidence below `0.70` for manual review.
- Queue samples when source expected label conflicts with Qwen's model label.
- Queue samples when the model places contact too close to the clip boundary; accepted clips need pre-contact context.
- Manually review at least 5%-10% of every production batch.
- If sampled error rate is high, rerun the whole batch with stricter prompts or recut clips.
