# Baseball Multimodal Dataset

This repository stores a multimodal baseball event dataset for machine learning research on hit-type classification.

The dataset focuses on:

- Audio: bat-ball contact sounds
- Video: MLB Film Room clips or similar source clips
- Temporal event segmentation
- Structured labels for supervised learning

The initial target classes are:

- Ground Ball
- Fly Ball

## Repository Structure

```text
baseball-multimodal-dataset/
├── dataset/
│   ├── ground_ball/
│   │   └── Zihan_Chai/
│   │       └── G_001/
│   │           ├── video.mp4
│   │           ├── audio.wav
│   │           ├── label.txt
│   │           ├── sample.csv
│   │           └── source.txt
│   └── fly_ball/
│       └── Zihan_Chai/
│           └── F_001/
│               ├── video.mp4
│               ├── audio.wav
│               ├── label.txt
│               ├── sample.csv
│               └── source.txt
├── docs/
│   ├── annotation_guideline.md
│   └── dataset_schema.md
└── README.md
```

## Sample Definition

Each sample is a temporal event segment, not a single timestamp. A segment should include:

- Pre-contact: pitch and bat preparation
- Contact moment: bat-ball collision
- Post-contact: early ball movement

Typical event durations:

- Ground Ball: about 4-6 seconds
- Fly Ball: about 5-7 seconds

## Naming Rules

- Ground Ball samples: `G_001`, `G_002`, ...
- Fly Ball samples: `F_001`, `F_002`, ...
- Annotator identity is encoded in the folder path, for example `dataset/ground_ball/Zihan_Chai/G_001/`.

## Dataset Principles

- No redundant metadata in CSV files
- Event-based temporal segmentation using `event_start` and `event_end`
- Traceability through `source.txt`
- Suitable for audio feature extraction, supervised learning, and future audio-video fusion research

See [docs/annotation_guideline.md](docs/annotation_guideline.md) and [docs/dataset_schema.md](docs/dataset_schema.md) for detailed rules.
