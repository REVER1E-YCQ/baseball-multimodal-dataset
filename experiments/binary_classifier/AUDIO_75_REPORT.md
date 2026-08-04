# Audio-Only Accuracy Study

## Objective

Raise source-grouped fly-ball versus ground-ball audio classification above 75% without using video frames.

The result must distinguish strict bat-contact audio from full audio context. Full clips contain the contact sound, field and crowd reactions, possible catch or bounce sounds, and commentary.

## Dataset and protocol

- 822 contact-verified samples: 386 fly balls and 436 ground balls.
- 802 source groups based on `source_id`, with URL fallback.
- Five-fold stratified source-group outer testing.
- Inner validation selects the audio feature combination and SVM parameters separately for every outer fold.
- No video feature is supplied to any model in this report.

## Window ablation

| Audio input | Model | Balanced accuracy |
|---|---|---:|
| 0.5 s contact window | Extra Trees | 64.6% |
| 1.0 s contact window | Extra Trees | 64.6% |
| 2.0 s early context | Extra Trees | 64.8% |
| 4.0 s extended context | Extra Trees | 71.8% |
| Full clip | RBF SVM | 73.7% |
| Full clip + 4 s + contact summary | RBF SVM, fixed exploratory configuration | 77.7% |
| Nested multiscale selection | RBF SVM | **76.5%** |

## Confirmatory result

Nested multiscale selection reaches:

- Accuracy: **76.5%**
- Balanced accuracy: **76.4%**
- Macro F1: **76.4%**
- ROC AUC: **0.844**
- Mean fold accuracy: **76.5%**, standard deviation 3.8 points

Outer-fold accuracies are 75.8%, 73.9%, 80.5%, 81.1%, and 71.3%. Three folds select full clip plus 4-second and contact summaries; two select full clip plus the 4-second summary.

The fixed three-scale configuration reaches 77.7%, but 76.5% is the primary number because nested selection accounts for choosing the feature combination.

## Source-bias checks

- Restricting evaluation to the 747 `Codex_Workstation` samples still yields 75.9% accuracy and 76.0% balanced accuracy for the fixed multiscale model.
- Source domain alone reaches 57.7% accuracy and 59.6% balanced accuracy.
- Source domain plus collector reaches 61.7% accuracy and 63.7% balanced accuracy.
- Adding the review route raises the metadata-only result to 70.9% accuracy. Review route is not an input to the audio model, but this result shows that curation provenance is label-correlated and must remain a documented limitation.

The audio model therefore exceeds simple source-domain and collector baselines, but long-context performance may still use broadcast style, commentary, crowd response, and other non-contact cues.

## Failed model

A small full-clip 2D audio CNN reaches only 62.0% accuracy and is unstable across folds. It is retained as a negative result. The dataset is currently better suited to regularized multiscale SVM features than training a CNN from scratch.

## Defensible conclusion

The project now has a pure-audio model above the requested 75% threshold: nested source-grouped accuracy is 76.5%. This is an **entire-audio-clip result**, not a claim that the 0.1-second bat-contact sound alone reaches 76.5%. Strict contact-window performance is currently approximately 65%.

For future independent confirmation, freeze the multiscale feature recipe and SVM search space before evaluating newly collected samples.
