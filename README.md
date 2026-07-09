# Baseball Hit Audio Dataset Workstation

This repository is a local workstation for building a baseball bat-ball contact sound dataset compatible with `REVER1E-YCQ/baseball-multimodal-dataset`.

Current pipeline:

1. Discover MLB official candidate videos into `manifests/sources_manifest.csv`.
2. Download source media into ignored `raw_sources/`.
3. Cut candidate clips and WAV audio into ignored `clips/pending/`, either manually or around detected audio peaks.
4. Label clips with Qwen-Omni through `scripts/qwen_omni_label.py`.
5. Materialize accepted labels into `dataset/<label>/<collector>/<sample_id>/`.
6. Run schema, media, contact-audio, and label QC gates.

Pilot command:

```powershell
$env:QWEN_API_KEY = "..."
.\scripts\run_pilot_batch.ps1 -StartDate 2026-07-01 -EndDate 2026-07-01 -SourceLimit 20 -LabelLimit 20 -Collector Your_Name
```

Do not commit real API keys. Large source media and candidate clips are intentionally ignored.
