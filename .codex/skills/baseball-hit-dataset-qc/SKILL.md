---
name: baseball-hit-dataset-qc
description: Validate, recut, retime, publish, or manually audit baseball bat-ball contact audio-video dataset batches under dataset/<label>/<collector>/<sample_id>. Use for ground_ball or fly_ball schema checks, long-video recuts, audio-first contact timing, Qwen/Omni visual verification, replay detection, batch release reports, or GitHub publication.
---

# Baseball Hit Dataset QC

## Required Gates

Each sample folder must contain `video.mp4`, `audio.wav`, `label.txt`, `sample.csv`, and `source.txt`. Folder label, file label, CSV label, and sample ID prefix must agree. `ffprobe` must read both media files and their durations must be close.

Use `validate_schema.py`, `validate_media.py`, `detect_contact_audio.py`, `audit_labels.py`, and a 5%-10% review sheet after materializing a batch.

## Audio-First Contact Timing

1. Decode the full clip audio and extract short transient candidates from energy and sample-difference peaks.
2. Use video only in a small neighborhood around each candidate to confirm that a live swing/contact occurs nearby and reject glove impacts, bounces, crowd peaks, commentary, cuts, replay audio, and slow motion.
3. Select one verified audio transient. Derive `event_start` and `event_end` in code around that transient, normally 0.05-0.15 seconds wide with at least 0.02 seconds on each side.
4. Treat audio as the authoritative clock. Record audio-video offset separately; never move the time label to match a delayed or early frame.
5. Reject or send to manual review when no distinct contact sound exists, the sound is masked, or it falls at the clip boundary.

Do not allow an AI model to directly set the final time window. It may choose or verify a local audio candidate only. Reject its result as `timing_inconsistent` when its stated contact time, selected candidate, audio evidence, and proposed interval disagree.

## Video And Semantic Rules

Use video for live-play verification and semantic labels only. Reject replay, slow motion, or clips without an observable live batted-ball sequence. For fly balls require `landing_zone`, `strength`, and `trajectory_type`; use `pending` only in the review queue, never in a released training sample.

For fly-ball recuts target about 2 seconds before contact and 10-12 seconds after. Mark clips that miss minimum context as `partial_context`; they can support review but never training.

## Batch Workflow

Process at most 250 samples, then produce a report with retained candidates, audio-time corrections, semantic changes, replay/slow-motion rejects, weak audio, partial-context clips, and manual-review items. After the batch is published, notify the user that the report is ready to inspect and immediately start the next batch unless the user has explicitly asked to stop.

Keep original media and metadata immutable until the batch decision is accepted. Store proposals and old-to-new fields in a separate audit manifest.

## Batch Publication

After a batch is reviewed and accepted, publish that batch's approved replacements to GitHub before starting the next batch.

For an unusable published sample, preserve its sample ID but clear the published sample directory and leave only `.gitkeep`. Record the ID, original path, reason, and batch in the batch audit manifest. Keep a local archival copy until all batches are complete; do not renumber samples or silently delete audit evidence.

Never publish AI proposals directly. Apply only audio-first validated changes or documented human decisions, regenerate the batch report, then commit and push the approved batch.

## References

Read `references/qc-checklist.md` for manual-review categories and `references/full-visual-audit.md` before replacing published labels.
