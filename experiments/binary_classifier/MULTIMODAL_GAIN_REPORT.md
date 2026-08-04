# Multimodal Gain Experiment

## Research question

Does verified bat-contact audio improve fly-ball versus ground-ball classification when visual trajectory evidence is still incomplete?

The experiment is framed as an early, low-latency decision task. Video input is limited to the contact frame and the frame approximately 0.45 seconds later. The visual feature contains both frames and their absolute difference. It does not use the later trajectory frames that make the class visually obvious.

## Protocol

- Dataset: 822 contact-verified samples, comprising 386 fly balls and 436 ground balls.
- Grouping: 802 source groups based on `source_id`, with source URL fallback.
- Evaluation: five-fold stratified source-group cross-validation.
- Model selection: each outer test fold has a separate inner validation fold.
- Audio: contact-centered Log-Mel representation minus a background window from the same clip.
- Video: contact frame, +0.45 second frame, and their absolute frame difference.
- Multimodal candidates: feature concatenation and late probability fusion.
- Fusion architecture, regularization, weights, and decision threshold are selected only inside each outer fold.
- Uncertainty: 10,000 class-stratified paired bootstrap resamples of out-of-fold predictions.

## Primary result

| Input | Out-of-fold balanced accuracy |
|---|---:|
| Early video only | 60.1% |
| Contact audio only | 63.0% |
| Audio + early video | **65.9%** |

- Multimodal minus video: **+5.8 percentage points**, 95% CI **+2.1 to +9.5**, probability of positive gain 99.9%.
- Multimodal minus audio: **+2.8 percentage points**, 95% CI **-0.18 to +5.86**, probability of positive gain 96.8%.
- Multimodal beats audio in four of five outer folds and beats video in four of five outer folds under architecture selection.

The multimodal point estimate is higher than both unimodal systems. The improvement over video is statistically supported by the paired interval. The improvement over audio is a strong directional result, but its two-sided 95% interval narrowly includes zero and should not yet be described as definitive.

## Time-window ablation

| Visual evidence available | Video | Audio | Multimodal | Gain over video |
|---|---:|---:|---:|---:|
| Contact frame only | 56.0% | 61.9% | 60.6% | +4.6 points |
| Contact to +0.45 s | 57.1% | 61.9% | 63.5% | +6.4 points |
| Contact to +0.45 s, including motion | 60.0% | 61.9% | 64.6% | +4.6 points |
| Contact to +1.35 s, including motion | 76.0% | 61.9% | 76.3% | +0.3 points |

Audio contributes most before the ball's visual trajectory is fully observable. Once 1.35 seconds of post-contact video and motion are available, the incremental audio gain becomes small and statistically inconclusive.

## Negative controls and failed alternatives

- Background-only audio reached approximately the same accuracy as the original contact Log-Mel baseline. This revealed broadcast and source confounding.
- Masking the central contact region reduced audio performance, showing that the contact transient contributes information but is not the only cue.
- Direct early feature concatenation with the full video representation did not improve the full-video baseline.
- A pretrained raw-waveform Wav2Vec2 encoder selected its shallow layer but did not improve the full-video classifier. Its audio-only balanced accuracy was 58.3%, and its multimodal result remained approximately equal to video alone.

These controls prevent the positive early-window result from being generalized to every fusion architecture or every video duration.

## Defensible conclusion

Verified contact audio provides complementary information for low-latency fly-ball versus ground-ball classification when only the contact frame and the first 0.45 seconds of video are available. The multimodal model reaches 65.9% balanced accuracy, exceeding video by 5.8 points and audio by 2.8 points under source-grouped nested evaluation. With longer post-contact video, visual trajectory dominates and the incremental audio contribution becomes negligible.

The next data collection cycle should increase the number of source-independent samples and preserve precise contact timing. That is more likely to make the audio-over-audio gain interval fully positive than additional tuning on the current 822 samples.
