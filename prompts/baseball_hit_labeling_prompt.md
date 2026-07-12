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
- The event interval must align with both the audible contact peak and the video frame where bat-ball contact occurs. If audio and video disagree, use uncertain or reject.
- For ground_ball region, divide the infield into four wedges from the home-plate viewpoint facing second base/outfield. Draw three boundary rays from home plate: one to the midpoint between third base and second base, one to second base, and one to the midpoint between first base and second base. From left to right:
  - 1: third-base line to the third/second midpoint boundary.
  - 2: third/second midpoint boundary to the home-to-second boundary.
  - 3: home-to-second boundary to the first/second midpoint boundary.
  - 4: first/second midpoint boundary to the first-base line.
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
