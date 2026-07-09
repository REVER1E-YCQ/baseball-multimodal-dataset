# Baseball Hit Audio Dataset Workstation

This repository is a local workstation for building a baseball bat-ball contact sound dataset compatible with `REVER1E-YCQ/baseball-multimodal-dataset`.

Current pipeline:

1. Discover MLB official candidate videos into `manifests/sources_manifest.csv`.
2. Download source media into ignored `raw_sources/`.
3. Cut candidate clips and WAV audio into ignored `clips/pending/`; use head-window clips first and audio-peak clips as a fallback.
4. Run local prefiltering with `scripts/prefilter_pending_clips.py` before spending Qwen tokens.
5. Label clips with Qwen-Omni through `scripts/qwen_omni_label.py`.
6. Materialize accepted labels into `dataset/<label>/<collector>/<sample_id>/`.
7. Run schema, media, contact-audio, and label QC gates.

Pilot command:

```powershell
$env:QWEN_API_KEY = "..."
.\scripts\run_pilot_batch.ps1 -StartDate 2026-07-01 -EndDate 2026-07-01 -SourceLimit 20 -LabelLimit 20 -Collector Your_Name
```

Do not commit real API keys. Large source media and candidate clips are intentionally ignored.

Cost controls:

```powershell
python scripts/summarize_qwen_usage.py
python scripts/prefilter_pending_clips.py
python scripts/qwen_omni_label.py --limit 5
```

If the cloud account reports billing issues, stop before `qwen_omni_label.py` and keep collecting/downloading/cutting candidates locally.
