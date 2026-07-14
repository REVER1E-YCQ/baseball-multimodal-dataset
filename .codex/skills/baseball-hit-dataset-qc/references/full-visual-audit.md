# Full Visual Audit

Use this policy when a batch must be certified rather than spot checked.

1. Run `qwen_review_dataset.py` so contact timing and semantic fields are judged in separate audio-visual passes.
2. For the time label, locate the bat-contact audio transient first and keep the interval centered on it. Record any visual offset instead of shifting the label to match a frame.
3. For ground-ball region, require the observed receiving or fielding location in the four home-plate-view wedges: third-base side=1, shortstop side=2, second-base side=3, first-base side=4. A fielder name alone is not evidence.
4. For bounce, require a visible receiving moment and a visible knee reference. Otherwise keep the sample in manual review.
5. Run adjudication for proposed corrections, low confidence, or disagreement.
6. Never apply model output directly. Reconcile accepted changes, regenerate the audit report, then run every required QC gate.
7. Do not publish while any accepted sample remains `incomplete` or has an unresolved required field.
8. Run `qwen_review_candidates.py` before materialization and require its visual pass in `materialize_dataset.py`.
9. Use `reconcile_qwen_dataset_review.py` to produce an old-to-new field report; apply only strict consensus or documented manual decisions.
10. Build `build_visual_audit_queue.py` for every unresolved sample and inspect both whole-clip and 25 fps event neighborhoods before writing a manual decision.
