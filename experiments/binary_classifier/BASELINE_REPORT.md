# Fly-Ball vs Ground-Ball Baseline Report

## Dataset and split

- 822 contact-verified samples: 386 fly balls and 436 ground balls.
- 802 source groups derived from `source_id`, with `video_url` as fallback.
- Fixed source-grouped split: 525 train, 132 validation, 165 test.
- Regularization is selected on validation data; the test set is used only for final metrics.

## Test results

| Input | Accuracy | Balanced accuracy | Macro F1 | ROC AUC |
|---|---:|---:|---:|---:|
| Majority class | 53.3% | 50.0% | 34.8% | 0.500 |
| Contact audio, Log-Mel | 60.0% | 59.8% | 59.8% | 0.597 |
| Contact audio, waveform | 54.5% | 54.4% | 54.4% | 0.514 |
| Contact-masked audio | 55.2% | 55.0% | 55.0% | 0.555 |
| Background audio control | 60.0% | 59.7% | 59.8% | 0.633 |
| Video appearance | 72.1% | 72.0% | 72.0% | 0.807 |
| Video motion | 73.3% | 73.5% | 73.3% | 0.801 |
| Video appearance + motion | **75.8%** | **75.6%** | **75.6%** | **0.832** |
| Contact audio + video | 73.3% | 73.2% | 73.2% | 0.800 |
| Background audio + video control | 76.4% | 76.5% | 76.3% | 0.836 |

## Paired uncertainty analysis

The 95% intervals use 10,000 class-stratified paired bootstrap resamples of the same test examples.

- Contact-audio balanced accuracy: 59.8% (95% CI 52.2%-67.0%).
- Video-combined balanced accuracy: 75.6% (95% CI 68.8%-81.8%).
- Video exceeds contact audio by 15.8 percentage points (95% CI 6.7-24.8), a reliable improvement on this test set.
- Contact-audio early fusion is 2.3 points below video alone (95% CI -8.0 to +3.2). There is no demonstrated multimodal gain yet.
- Background-audio fusion is 0.9 points above video alone (95% CI -5.3 to +7.1). This is statistically inconclusive and is treated as a leakage control, not a useful audio contribution.

## Interpretation

The current visual representation carries substantially more information about fly-ball versus ground-ball outcome than the current audio representation. Removing the central contact region lowers audio balanced accuracy from 59.8% to 55.0%, so the contact transient contributes some information. However, background-only audio performs similarly to contact audio, showing that broadcast, commentary, or collection-source cues remain a major confounder.

At this stage, the evidence does **not** support the claim that adding this audio representation improves a video classifier. The next audio experiment should use a pretrained general audio-event encoder and repeat source-grouped evaluation across multiple seeds. A multimodal improvement should only be accepted if it exceeds video-only performance consistently and also beats the background-audio fusion control.
