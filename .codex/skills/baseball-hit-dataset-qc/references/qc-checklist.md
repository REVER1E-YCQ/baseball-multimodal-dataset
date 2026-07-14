# Baseball Hit Dataset QC Checklist

Use these categories in manual review reports:

- `pass`: Label and cut are usable.
- `wrong_label`: Ground/fly class or subclass fields are wrong.
- `bad_audio`: Contact sound is missing, masked, clipped, or too noisy.
- `bad_cut`: Clip misses pre-contact context, contact, or early ball movement.
- `source_issue`: Source URL, title, rights note, or traceability is missing or suspicious.
- `audio_video_offset`: the valid bat-contact audio transient and the visual collision are measurably offset. Keep the event time audio-centered and record the offset; this is not a reason to move the time label to the video frame.

Authoritative fields:

- `event_start` / `event_end`: audio first. Center a short interval on the bat-contact transient in `audio.wav`.
- `region`, `bounce`, `strength`, `landing_zone`, `trajectory_type`, and class: video first.
- Ground-ball regions: at the first fielding/control frame, project the BALL onto a top-down infield fan and divide the fair infield from third-base line to first-base line into four equal left-to-right sectors (`1`, `2`, `3`, `4`). Do not use the fielder's nominal position, identity, ball path, later throw, or camera orientation as a proxy.

Recommended report fields:

```csv
sample_path,label,event_start,event_end,manual_result,reviewer,notes
```

Escalate a batch when sampled error rate is high enough that isolated fixes are less reliable than recutting or relabeling the whole batch.
