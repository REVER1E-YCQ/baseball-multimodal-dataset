param(
  [Parameter(Mandatory = $true)]
  [string]$StartDate,

  [Parameter(Mandatory = $true)]
  [string]$EndDate,

  [int]$SourceLimit = 20,
  [int]$LabelLimit = 20,
  [string]$Collector = "Codex_Workstation"
)

$ErrorActionPreference = "Stop"

if ($SourceLimit -lt 1) {
  throw "SourceLimit must be at least 1. Use a small positive number for pilot batches."
}

if ($LabelLimit -lt 1) {
  throw "LabelLimit must be at least 1. Use a small positive number for pilot batches."
}

python scripts/collect_mlb_sources.py --start-date $StartDate --end-date $EndDate --limit $SourceLimit
python scripts/download_sources.py

python scripts/auto_extract_candidates.py --limit $SourceLimit --candidates-per-source 1

python scripts/qwen_omni_label.py --limit $LabelLimit
python scripts/refine_qwen_events.py
python scripts/audit_qwen_labels.py --labels reports/qwen_labels_refined.jsonl
python scripts/materialize_dataset.py --labels reports/qwen_labels_refined.jsonl --collector $Collector --require-audit-pass
python scripts/validate_schema.py
python scripts/validate_media.py
python scripts/detect_contact_audio.py
python scripts/audit_labels.py
python scripts/build_review_sheet.py --sample-rate 0.10
