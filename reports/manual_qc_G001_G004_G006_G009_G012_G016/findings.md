# Manual QC Findings: G001/G004/G006/G009/G012/G016

Data collection and model labeling are paused. This review used local video contact sheets, marked waveforms, marked spectrograms, source metadata, and saved Qwen evidence.

Region mapping has now been defined for this project: from the home-plate viewpoint facing second base/outfield, region `1` is the third-base side wedge, `2` is the left-middle/shortstop wedge up to the home-to-second boundary, `3` is the right-middle/second-base wedge, and `4` is the first-base side wedge.

Bounce mapping has now been defined for this project: for a ground ball, `bounce=yes` means the ball is at or below the receiving fielder's knee height when fielded; `bounce=no` means the receiving height is above the fielder's knee.

| sample | current label | evidence | QC decision |
| --- | --- | --- | --- |
| G_001 | region=4, bounce=yes, event=1.400-1.600 | Audio peak is inside the event window; video shows contact around 1.48-1.58s. Source/Qwen evidence says grounder toward second/right side. | I do not see a clear timing or bounce error. Region still needs user-specific issue notes because second/right-side contact may be region 3 or 4 depending the exact lane. |
| G_004 | region=2, bounce=yes, event=1.420-1.520 | Audio RMS peak is at 1.47s and video contact is around 1.45-1.55s. Source says Tyler Callihan/Jose Tena ground ball. | I do not confirm a major timing mismatch; at most the end could be slightly extended. Region should be relabeled using the new four-wedge mapping. |
| G_006 | region=4, bounce=yes, event=2.460-2.560 | Audio/video contact aligns at about 2.52s. Source says ground ball to right field. | Timing is OK. Bounce requires a formal definition; under the user's review standard, mark for relabel to bounce=no or manual adjudication. |
| G_009 | region=3, bounce=yes, event=2.220-2.320 | Video contact is around 1.0s; audio strongest transient is around 1.02s. Current event is after the camera cut/in play. | User is correct: timing is wrong. Region and bounce should be discarded and relabeled from the actual contact window. |
| G_012 | region=3, bounce=yes, event=1.980-2.080 | Audio/video contact aligns at about 2.04s. Source/Qwen evidence says ground ball to third. | Timing is OK. Under the new mapping, a third-base-side grounder should be region 1, so current region=3 is wrong or at least needs manual relabel. |
| G_016 | region=4, bounce=yes, event=1.200-1.300 | Audio/video contact aligns at about 1.26s. Source/Qwen evidence says ground ball to shortstop/CJ Abrams. | Timing is OK. Under the new mapping, a shortstop-side grounder is likely region 2, so current region=4 is wrong or at least needs manual relabel. |

Generated evidence assets live in this folder:

- `G_*/video_sheet.png`
- `G_*/video_zoom_event.png`
- `G_*/waveform_marked.png`
- `G_*/spectrum_marked.png`
- `audio_peak_summary.csv`
