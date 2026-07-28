---
name: baseball-hit-dataset-qc
description: Audit and repair baseball bat-ball contact datasets, especially fly_ball batches on GitHub main. Use for audio-first contact validation, source recovery and longer recuts, Qwen visual confirmation of local audio candidates, batch reports, and publish gates.
---

# Baseball Hit Dataset QC

Use `origin/main` as the authoritative dataset state. Preserve its collector and
sample path exactly. Treat four-digit `Codex_Workstation/F_####` IDs and
three-digit contributor IDs as separate namespaces.

## Full Audit Before Editing

1. Fetch `origin/main` and work from a clean branch or worktree based on it.
2. Inventory every `dataset/fly_ball/<collector>/<sample_id>` directory.
3. Check all five required files: `video.mp4`, `audio.wav`, `label.txt`,
   `sample.csv`, and `source.txt`.
4. Analyze `audio.wav` around the existing `event_start` and `event_end`.
5. Record local transient candidates, the candidate nearest the annotation,
   audio confidence, clip duration, pre-contact context, post-contact context,
   source URL, source path, and `clip_start_time`.
6. Classify each row as a direct-use candidate or needs editing. Separate at
   least: wrong timestamp, missing or masked contact audio, clip too short,
   replay or slow motion, semantic uncertainty, and unreadable media.
7. Write the complete audit CSV and a count summary before changing any sample.

The audit is non-destructive. A direct-use candidate is still subject to a
video spot check; it is not a guaranteed training sample merely because a loud
sound occurs near the annotation.

## Repair Batches

Split only the needs-editing rows into four batches while retaining main order.
Each queue row must include its exact main-relative path and original metadata.

For each batch:

1. Keep the original sample immutable until a replacement passes all gates.
2. If context is short or contact audio is absent, locate the original source.
3. Convert sample-relative contact time to source time with:
   `source_contact = clip_start_time + sample_contact`.
4. Recut approximately 2 seconds before contact and 10 to 12 seconds after
   contact when the source permits it. Record partial context explicitly.
5. Run local audio analysis on the repaired clip and produce a finite list of
   timestamped contact candidates.
6. Give Qwen the repaired audio-video clip and the supplied candidate list.
   Qwen may confirm or reject candidates and judge live-play/replay semantics;
   it must not invent an unrestricted final timestamp.
7. Bind the accepted result to one supplied audio candidate. Derive a narrow
   0.05 to 0.15 second event interval in code.
8. Create a second, candidate-centered excerpt of roughly 1.4 seconds. Give it
   to an independent Qwen model without trusting the first response. Require a
   visible live pitch, batter swing, contact/follow-through, and a matching bat
   sound near the center candidate.
9. Reject the second pass when it shows only ball flight, an outfielder, catch,
   runner, celebration, replay, or generic evidence. Restore an already
   published failed sample to its pre-batch version before further recovery.
10. Preserve the source URL, source path, source offset, before/after values,
   model response, and decision evidence.

## Hard Rules

- Audio is the authoritative clock. Video confirms that a live bat-ball action
  occurs near the audio candidate.
- Do not accept crowd noise, commentary, glove pops, edits, or replay audio as
  contact merely because they are high-energy sounds.
- Do not publish a first-pass Qwen acceptance without the independent,
  candidate-centered contact gate.
- Do not accept placeholder or generic model evidence. Evidence must describe
  the visible batter/pitch/swing/contact and the specific candidate sound.
- Do not let the contact-timing model automatically change trajectory metadata;
  route proposed fly, line-drive, or pop-fly changes to a separate review.
- Do not replace unresolved data with an empty directory or placeholder.
- Do not silently drop a sample. Keep it unchanged and mark it unresolved until
  source recovery and review are complete.
- Do not publish a partial-context replacement as final without explicit review.
- Do not publish work created from a legacy branch or remapped sample number.
- Never expose API keys in tracked files, logs, reports, prompts, or commits.

## Batch Publish Gate

Before publishing a batch:

1. Verify schema and all required files.
2. Verify audio and video are readable and synchronized.
3. Verify the final event interval is bound to a local audio candidate.
4. Verify both Qwen passes place live visual contact near that candidate, and
   verify that the second pass used a different model from the first.
5. Verify fly-ball semantics and reject replay-only footage.
6. Reconcile every queue row to exactly one outcome: replaced, metadata-only
   correction, unchanged unresolved, or confirmed unusable with evidence.
7. Generate a detailed CSV and Markdown report listing every changed sample,
   before/after timestamps, recut lengths, source offsets, error categories,
   unresolved items, and validation results.
8. Fetch `origin/main`, integrate any new remote commits without force pushing,
   and publish the verified batch directly to `main`.
9. Notify the user that the batch is available for inspection, then immediately
   start the next batch.

See [workflow.md](references/workflow.md) for required report fields and stop
conditions.
