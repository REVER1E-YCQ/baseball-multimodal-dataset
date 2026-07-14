You are auditing exactly one field: a ground ball's absolute receiving region.

Return only strict JSON.  Do not use any original CSV label, source text,
scoreboard, commentary, player names, hit direction, ball-path direction, or
the camera's temporary left/right orientation.  Those inputs are deliberately
withheld because they are not evidence for this task.

Watch the video frame by frame and locate the first moment when the batted
ball is controlled. This is the first fielding/control event, not a later
throw recipient, relay, baseman covering a bag, or tag. At that exact frame,
place the BALL (not the fielder) onto a top-down infield diamond. Divide the
fair infield fan from the third-base foul line to the first-base foul line into
four equal left-to-right sectors:

- leftmost quarter = region 1
- left-middle quarter = region 2
- right-middle quarter = region 3
- rightmost quarter = region 4

Region is the ball's absolute infield location at first control. Do not use
the fielder's nominal position, player identity, where the ball was hit,
subsequent throws, or screen left/right as a substitute. If the ball at first
control cannot be located on the top-down infield plane, return `review` and
leave `region` null. Report the approximate clip timestamp and a concise
description of the ball's absolute location.

JSON schema:
{
  "decision": "pass|review",
  "confidence": 0.0,
  "first_control_time_seconds": 0.0,
  "ball_absolute_position": "leftmost|left_middle|right_middle|rightmost|unresolved",
  "region": 1,
  "receiving_moment_visible": true,
  "evidence": "short frame-level description"
}
