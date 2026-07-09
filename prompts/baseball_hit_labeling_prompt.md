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
   - region: integer 1-4
   - strength: low, medium, or high
   - bounce: yes or no
4. If fly_ball, provide:
   - landing_zone: integer 1-9
   - strength: low, medium, or high
   - trajectory_type: fly, line_drive, or pop_fly
5. Estimate the audible contact event interval within the clip:
   - event_start: seconds from clip start
   - event_end: seconds from clip start

Rules:
- Use reject if the contact sound is missing, heavily masked, replay-only, or the clip does not show the batted-ball event.
- Use uncertain if evidence exists but the hit type or timing cannot be determined with confidence.
- The event interval should bracket the bat-ball collision itself, not the whole play.
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
    "region": 1,
    "strength": "low|medium|high",
    "bounce": "yes|no"
  },
  "fly_ball": {
    "landing_zone": 1,
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

