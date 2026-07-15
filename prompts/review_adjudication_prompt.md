You are the final challenger for a baseball dataset review. Return only strict JSON.

Rewatch the full video and challenge both prior audit results supplied after this prompt. Resolve disagreements from direct audio-video evidence, not majority vote. In particular, reject a timing based on a later glove/bounce/commentary transient, and do not assign bounce unless the fielding height can be compared with the receiver's knee.

Non-negotiable definitions:
- `event_start` and `event_end` bracket ONLY the instantaneous bat-ball collision, normally 0.05-0.20 seconds wide. They never span the pitch, ball flight, bounce, catch, or whole play.
- Use `accept_correction` when prior timing or fields are wrong but the sample has a clear usable bat-ball contact. Do not use `reject` merely because a correction is needed.
- Use `reject` only when the sample itself is unusable: contact is absent/inaudible, replay-only, irreconcilably out of sync, or the class cannot be resolved.
- A ground_ball first contacts the ground in the infield and travels primarily along/near the ground through the infield. A line_drive remains fly_ball even if it later lands or bounces in the outfield.
- Region geometry is immutable regardless of camera orientation:
  `3B line | 1 | 3B-2B midpoint | 2 | second base | 3 | 1B-2B midpoint | 4 | 1B line`.
  Third-base-side evidence can only produce 1/2; first-base-side evidence can only produce 3/4.
- Region is the BALL'S absolute position at first fielding/control, or—only when no defender
  controls it in the clip—its last clear locatable position in the fair infield. Never use the
  fielder, player identity, ball path, later throw, or screen orientation as a tie-breaker. If the
  ball cannot be located on the top-down infield plane, leave region unresolved for review.
- The observed collision time must be inside the event interval with time on both sides. Pitch
  release and swing onset are context, never event boundaries.

Only use `accept_current` or `accept_correction` when every accepted field has visible/audible evidence and confidence is at least 0.85. Otherwise use `manual_review` or `reject`.

JSON schema:
{
  "decision": "accept_current|accept_correction|manual_review|reject",
  "confidence": 0.0,
  "contact_visible": true,
  "contact_audible": true,
  "audio_video_aligned": true,
  "label": "ground_ball|fly_ball|reject",
  "event_start": 0.0,
  "event_end": 0.0,
  "strength": "low|medium|high|unverified",
  "region": 1,
  "bounce": "yes|no|unverified",
  "landing_zone": 1,
  "trajectory_type": "fly|line_drive|pop_fly|unverified",
  "timing_evidence": "short direct evidence",
  "field_evidence": "short direct evidence",
  "unresolved_fields": [],
  "failure_reason": ""
}
