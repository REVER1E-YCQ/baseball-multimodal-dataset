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
3. Cut candidate clips with `scripts/extract_candidates.py`.
4. Label pending clips with `scripts/qwen_omni_label.py`.
5. Materialize accepted labels into `dataset/` with `scripts/materialize_dataset.py`.
6. Run validation:
   - `scripts/validate_schema.py`
   - `scripts/validate_media.py`
   - `scripts/detect_contact_audio.py`
   - `scripts/audit_labels.py`
7. Generate an HTML review sheet with `scripts/build_review_sheet.py`.
8. Commit only reviewed batches.

Pilot command:

```powershell
$env:QWEN_API_KEY = "..."
.\scripts\run_pilot_batch.ps1 -StartDate 2026-07-01 -EndDate 2026-07-01 -SourceLimit 20 -LabelLimit 20 -Collector Your_Name
```

## Source Policy

Store source URLs and play descriptions for traceability. Use public MLB Film Room or similarly accessible clips for internal research. Before public redistribution, confirm rights for media files or publish only metadata/features that are allowed by the source terms.

## Scale Plan

Use a small pilot first:

- Pilot: 10-20 samples.
- Batch 1: 100 samples.
- Production batches: 300-500 samples each.
- Final target: about 2000 accepted samples after rejects and audit failures.
