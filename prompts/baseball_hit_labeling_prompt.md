You are labeling short baseball broadcast clips for a multimodal bat-ball contact sound dataset.

Return only strict JSON. Do not include Markdown.

Task:
1. Decide whether the clip contains a clear bat-ball contact sound.
2. Use both audio and video evidence to classify the hit as one of:
   - ground_ball
   - fly_ball
   - reject
   - uncertain
3. If ground_ball, provide:
   - region: null (position collection is deferred until the dataset reaches 2,000 samples)
   - strength: low, medium, or high
   - bounce: yes or no
4. If fly_ball, provide:
   - landing_zone: null (position collection is deferred until the dataset reaches 2,000 samples)
   - strength: low, medium, or high
   - trajectory_type: fly, line_drive, or pop_fly
5. Estimate the audible contact event interval within the clip:
   - event_start: seconds from clip start
   - event_end: seconds from clip start

Priority order for this production pass:
1. Clear audible bat-ball contact and its tight audio time interval.
2. Correct primary class: ground_ball versus fly_ball.
3. Strength, bounce, and trajectory fields are best-effort only. Do not spend extra analysis rounds on them and do not reject an otherwise valid sample solely because one is approximate.
4. Do not analyze, infer, or estimate ground-ball region or fly-ball landing zone in this collection pass. Return null for those fields.

Rules:
- Use reject if the contact sound is missing, heavily masked, replay-only, or the clip does not show the batted-ball event.
- Use uncertain if evidence exists but the hit type or timing cannot be determined with confidence.
- The event interval should bracket the bat-ball collision itself, not the whole play.
- The event interval is audio-first: tightly bracket the bat-contact sound, normally 0.05-0.20 seconds. If video and audio are offset, preserve the audio-centred time rather than moving it to match the picture. Report the offset in `audio_evidence`; do not reject a usable clip solely because of offset.
- Do not infer bounce=yes only because the hit is a ground_ball. For ground_ball, mark bounce=yes only when the ball is at or below the receiving fielder's knee height when fielded. Mark bounce=no when the receiving height is above the fielder's knee. If the receiving fielder or receiving height cannot be judged, lower confidence or use uncertain.
- Prefer conservative labels. A bad sample is worse than a rejected sample.
- Do not invent source metadata.

JSON schema:
{
  "label": "ground_ball|fly_ball|reject|uncertain",
  "confidence": 0.0,
  "contact_sound_clear": true,
  "event_start": 0.0,
  "event_end": 0.0,
  "ground_ball": {
    "region": null,
    "strength": "low|medium|high",
    "bounce": "yes|no"
  },
  "fly_ball": {
    "landing_zone": null,
    "strength": "low|medium|high",
    "trajectory_type": "fly|line_drive|pop_fly"
  },
  "audio_quality": {
    "contact_peak": "clear|masked|absent",
    "commentary_overlap": "none|minor|major",
    "crowd_noise": "low|medium|high"
  },
  "video_evidence": "short evidence sentence",
  "audio_evidence": "short evidence sentence",
  "failure_reason": ""
}
