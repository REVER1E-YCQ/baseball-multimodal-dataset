---
name: baseball-hit-dataset-qc
description: Validate baseball bat-ball contact audio-video dataset batches that follow dataset/<label>/<collector>/<sample_id>/ with video.mp4, audio.wav, label.txt, sample.csv, and source.txt. Use when checking schema compliance, media readability, Qwen/Omni labels, contact-sound timing, source traceability, or manual-review readiness for ground_ball and fly_ball samples.
---

# Baseball Hit Dataset QC

## Quick Start

Run these gates from the repository root after materializing a batch:

```powershell
python scripts/validate_schema.py
python scripts/validate_media.py
python scripts/detect_contact_audio.py
python scripts/audit_labels.py
python scripts/build_review_sheet.py --sample-rate 0.10
```

Treat any failure as a batch issue until the failing samples are fixed, rejected, or moved back to `clips/rejected/`.

## Required Gates

1. Schema: each sample folder must contain `video.mp4`, `audio.wav`, `label.txt`, `sample.csv`, and `source.txt`.
2. Label path consistency: folder label, `label.txt`, CSV label, and sample ID prefix must agree.
3. Media: `ffprobe` must read video and audio; durations must be close.
4. Contact audio: the strongest short transient should be near `event_start` and `event_end`.
5. Label values: ground balls require `region`, `strength`, and `bounce`; fly balls require `landing_zone`, `strength`, and `trajectory_type`.
6. Source traceability: `source.txt` must include `video_title:` and `video_url:`.

## Decision Rules

- Reject samples where bat-ball contact is absent, masked by commentary, or located at the clip boundary.
- Send samples to manual review when model confidence is below 0.70, hit type conflicts with source text, or audio transient checks fail.
- Keep `event_start` and `event_end` tight around contact, normally under 0.200 seconds.
- Build a review sheet for at least 5%-10% of each production batch.

## References

Read `references/qc-checklist.md` when preparing a human audit handoff or interpreting failure categories.

