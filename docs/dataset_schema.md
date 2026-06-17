# Dataset Schema

## Folder Schema

```text
dataset/<label>/<annotator>/<sample_id>/
```

Examples:

```text
dataset/ground_ball/Zihan_Chai/G_001/
dataset/fly_ball/Zihan_Chai/F_001/
```

Each sample folder must contain:

```text
video.mp4
audio.wav
label.txt
sample.csv
source.txt
```

## Ground Ball CSV Schema

```csv
sample_id,label,region,strength,bounce,event_start,event_end
```

Fields:

- `sample_id`: sample identifier, for example `G_001`
- `label`: must be `ground_ball`
- `region`: integer field zone from 1 to 4
- `strength`: one of `low`, `medium`, `high`
- `bounce`: one of `yes`, `no`
- `event_start`: event start time in seconds
- `event_end`: event end time in seconds

## Fly Ball CSV Schema

```csv
sample_id,label,landing_zone,strength,trajectory_type,event_start,event_end
```

Fields:

- `sample_id`: sample identifier, for example `F_001`
- `label`: must be `fly_ball`
- `landing_zone`: integer field position from 1 to 9
- `strength`: one of `low`, `medium`, `high`
- `trajectory_type`: one of `fly`, `line_drive`, `pop_fly`
- `event_start`: event start time in seconds
- `event_end`: event end time in seconds

## Source File Schema

```text
video_title: original MLB play description
video_url: original video link
```

## Design Notes

Annotator metadata is stored in the folder path rather than repeated in CSV files. CSV files only store sample-level labels and event timing values needed by machine learning pipelines.
