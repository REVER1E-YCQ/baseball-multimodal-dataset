# Fly Ball Reclean Workflow

## Audit CSV Fields

The full audit CSV must identify the exact main path and include:

- global order, collector, sample ID, and digit width
- current event start and end
- annotated transient time and score
- nearest and strongest local audio candidates
- candidate list as structured JSON
- audio assessment and primary error category
- video/audio duration and pre/post-contact context
- source URL, source path, clip start/end, and local source availability
- required repair actions

## Context Targets

- `fly` and `pop_fly`: at least about 1 second before contact and 8 seconds
  after contact in the final clip; prefer 2 seconds before and 10 to 12 seconds
  after. Allow 0.05 seconds of timestamp/frame rounding tolerance at the gate.
- `line_drive`: at least 0.8 seconds before and 4 seconds after contact.
- If the source ends before the target, record `partial_context` and route the
  sample to review instead of silently accepting it.

## Qwen Contract

Provide Qwen with numbered local audio candidates. Require:

- selected candidate index
- selected candidate timestamp copied exactly
- whether contact sound is audible
- whether live visual bat-ball contact occurs near the candidate
- whether footage is replay or slow motion
- trajectory and confidence
- numeric visual contact time, within 0.35 seconds of the selected audio candidate
- concise evidence

Reject the response if the timestamp is not one of the supplied candidates or
if audible and visual contact are not both confirmed. Program code, not the
model, writes the final interval.

`full_play_visible` is the final visual completeness gate. A recut that misses
the preferred numeric context target may still pass when Qwen or a human
confirms that the complete pitch, contact, and fly-ball result are visible.

## Per-Batch Report

The batch CSV and Markdown report must state:

- queue size and completed count
- unchanged, recut, retimed, metadata-only, unresolved, and confirmed unusable
  counts
- complete-context and partial-context recut counts
- replay/slow-motion and semantic correction counts
- source recovery success and failure counts
- every changed sample with before/after timestamp and duration
- every unresolved sample and the exact reason
- validation command results and the Git commit published to `main`

## Stop Conditions

Do not publish when:

- any replacement lacks a required file
- a final timestamp is not bound to a local audio candidate
- the candidate lacks matching live visual contact evidence
- a queue row disappears from reconciliation
- an unresolved sample was replaced by an empty folder
- `origin/main` advanced and the batch has not been reconciled with it
