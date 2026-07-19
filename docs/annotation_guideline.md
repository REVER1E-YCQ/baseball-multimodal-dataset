# Annotation Guideline

## Annotation Tool

Use Audacity for temporal annotation of bat-ball contact events.

## Core Rule

Do not annotate each event as a single timestamp. Each event must be annotated with:

```text
event_start_time
event_end_time
```

The interval should capture the audible contact event and the immediate surrounding context needed for downstream learning.

## Event Segment Definition

Each sample represents one temporal event segment containing:

- Pre-contact: pitch delivery and swing preparation
- Contact moment: bat-ball collision
- Post-contact: initial ball movement and immediate reaction

Typical segment duration:

- Ground Ball: about 4-6 seconds
- Fly Ball: about 5-7 seconds

## Ground Ball Annotation

Ground ball samples use:

- `region`: integer field zone from 1 to 4
- `strength`: one of `low`, `medium`, `high`
- `bounce`: one of `yes`, `no`
- `event_start`: event start time in seconds
- `event_end`: event end time in seconds

Example:

```csv
sample_id,label,region,strength,bounce,event_start,event_end
G_001,ground_ball,1,medium,yes,3.307,3.313
```

## Fly Ball Annotation

Fly ball samples use:

- `landing_zone`: integer field position from 1 to 9
- `strength`: one of `low`, `medium`, `high`
- `trajectory_type`: one of `fly`, `line_drive`, `pop_fly`
- `event_start`: event start time in seconds
- `event_end`: event end time in seconds

Example:

```csv
sample_id,label,landing_zone,strength,trajectory_type,event_start,event_end
F_001,fly_ball,8,high,fly,2.305,2.320
```

## Source Tracking

Every sample folder must include `source.txt` with:

```text
video_title: original MLB play description
video_url: original video link
```
