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

python scripts/collect_mlb_sources.py --start-date $StartDate --end-date $EndDate --limit $SourceLimit
python scripts/download_sources.py

$sources = Import-Csv manifests/sources_manifest.csv | Where-Object { $_.status -eq "downloaded" -and $_.local_path }
foreach ($source in $sources | Select-Object -Last $SourceLimit) {
  $duration = if ($source.expected_label -eq "fly_ball") { 7.0 } else { 6.0 }
  python scripts/extract_candidates.py `
    --source-id $source.source_id `
    --source-path $source.local_path `
    --start 0 `
    --end $duration `
    --expected-label $source.expected_label
}

python scripts/qwen_omni_label.py --limit $LabelLimit
python scripts/materialize_dataset.py --collector $Collector
python scripts/validate_schema.py
python scripts/validate_media.py
python scripts/detect_contact_audio.py
python scripts/audit_labels.py
python scripts/build_review_sheet.py --sample-rate 0.10

