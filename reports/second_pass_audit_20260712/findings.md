# Second Pass Audit

- Checked samples: 403
- Samples changed: 4
- Samples still needing visual review: 339

## Changed Samples

| sample | changed_fields | old -> new | reason |
| --- | --- | --- | --- |
| G_006 | bounce | bounce: yes -> no | user_manual_qc: receiving height exceeds knee-height bounce standard |
| G_009 | region,bounce,event_start,event_end | region: 3 -> 1; bounce: yes -> no; event_start: 2.220 -> 0.970; event_end: 2.320 -> 1.070 | manual_qc: true contact is about 1.02s on third-base-side play |
| G_012 | region | region: 3 -> 1 | source/qwen evidence says third-base-side ground ball |
| G_016 | region | region: 4 -> 2 | source/qwen evidence says shortstop-side ground ball |

## Review Policy

- `region` can be suggested from source/Qwen evidence, but field-side ambiguity still requires visual review.
- `bounce` requires the receiving fielder's knee-height standard and cannot be proven from CSV ranges alone.
- `time_status=pass_audio_only_needs_visual_confirmation` means the audio check passed but visual contact still has not been certified.
