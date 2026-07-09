# Baseball Hit Audio Dataset Workflow

This workstation builds a dataset compatible with `REVER1E-YCQ/baseball-multimodal-dataset`.

## Target Schema

Final samples live at:

```text
dataset/<label>/<collector>/<sample_id>/
```

Each sample must contain:

```text
video.mp4
audio.wav
label.txt
sample.csv
source.txt
```

Ground ball CSV:

```csv
sample_id,label,region,strength,bounce,event_start,event_end
```

Fly ball CSV:

```csv
sample_id,label,landing_zone,strength,trajectory_type,event_start,event_end
```

## Batch Flow

1. Add source rows to `manifests/sources_manifest.csv`.
2. Download sources into `raw_sources/` with `scripts/download_sources.py`.
3. Cut candidate clips. Prefer `scripts/extract_head_candidates.py` for MLB highlight clips, then use `scripts/auto_extract_candidates.py` as a secondary pass when head windows miss contact.
4. Run local prefiltering with `scripts/prefilter_pending_clips.py` to reduce paid multimodal calls.
5. Check historical model usage with `scripts/summarize_qwen_usage.py`.
6. Label pending clips with `scripts/qwen_omni_label.py`.
7. Refine Qwen contact timings with `scripts/refine_qwen_events.py`.
8. Audit Qwen labels with `scripts/audit_qwen_labels.py --labels reports/qwen_labels_refined.jsonl`.
9. Materialize accepted labels into `dataset/` with `scripts/materialize_dataset.py --labels reports/qwen_labels_refined.jsonl --require-audit-pass`.
10. Run validation:
   - `scripts/validate_schema.py`
   - `scripts/validate_media.py`
   - `scripts/detect_contact_audio.py`
   - `scripts/audit_labels.py`
11. Generate an HTML review sheet with `scripts/build_review_sheet.py`.
12. Commit only reviewed batches.

Pilot command:

```powershell
$env:QWEN_API_KEY = "..."
.\scripts\run_pilot_batch.ps1 -StartDate 2026-07-01 -EndDate 2026-07-01 -SourceLimit 20 -LabelLimit 20 -Collector Your_Name
```

Manual staged pilot:

```powershell
python scripts/collect_mlb_sources.py --start-date 2026-07-01 --end-date 2026-07-01 --limit 20
python scripts/download_sources.py
python scripts/extract_head_candidates.py --limit 20
python scripts/auto_extract_candidates.py --limit 20 --candidates-per-source 1
python scripts/prefilter_pending_clips.py
python scripts/summarize_qwen_usage.py
python scripts/qwen_omni_label.py --limit 20
python scripts/refine_qwen_events.py
python scripts/audit_qwen_labels.py --labels reports/qwen_labels_refined.jsonl
python scripts/materialize_dataset.py --labels reports/qwen_labels_refined.jsonl --collector Your_Name --require-audit-pass
```

For larger backfills, prefer the resumable batch collector. It writes the manifest after each game and deduplicates by both `source_id` and `source_url`:

```powershell
python scripts/collect_mlb_sources_batch.py --start-date 2026-05-01 --end-date 2026-05-31 --target-new 100
```

## Source Policy

Store source URLs and play descriptions for traceability. Use public MLB Film Room or similarly accessible clips for internal research. Before public redistribution, confirm rights for media files or publish only metadata/features that are allowed by the source terms.

## Scale Plan

Use a small pilot first:

- Pilot: 10-20 samples.
- Batch 1: 100 samples.
- Production batches: 300-500 samples each.
- Final target: about 2000 accepted samples after rejects and audit failures.
