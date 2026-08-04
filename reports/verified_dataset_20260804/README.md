# Verified Dataset Branch

This branch is based on `origin/main` and its `dataset/` tree contains only samples accepted by either human review or the completed local first-pass contact review.

## Counts

- Total: 822
- Fly ball: 386
- Ground ball: 436
- Human reviewed: 498
- Local first-pass direct: 324
- Human timing corrections materialized: 22

## Inclusion Contract

- Every sample has `video.mp4`, `audio.wav`, `label.txt`, `sample.csv`, and `source.txt`.
- Folder, `label.txt`, and `sample.csv` class identities agree.
- Event intervals are valid and no longer than 0.2 seconds.
- Human rows marked usable are included; explicit incorrect timing rows are recentered to 0.1 seconds around the supplied contact time.
- Samples marked uncertain or needing repair in the local first pass are excluded.

## Scope

The verified decision concerns bat-ball contact presence, timing usability, and the binary `fly_ball` / `ground_ball` label needed for classifier experiments. Some secondary semantic fields may remain `pending`; see `SECONDARY_FIELD_WARNINGS.txt`.
