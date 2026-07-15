$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo
if (-not $env:QWEN_API_KEY) { throw 'QWEN_API_KEY must be set in the launching process.' }
$completed = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
$outputPath = 'reports/qwen_ground_region_video_only_audit_20260714.jsonl'
if (Test-Path $outputPath) {
    Get-Content $outputPath | ForEach-Object {
        try {
            $record = $_ | ConvertFrom-Json
            if ($record.stage -eq 'semantics' -and $record.sample_id) { [void]$completed.Add($record.sample_id) }
        } catch { }
    }
}
$ids = Get-ChildItem 'dataset/ground_ball' -Recurse -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'sample.csv') } | ForEach-Object Name | Where-Object { -not $completed.Contains($_) } | Sort-Object
$args = @('scripts/qwen_review_dataset.py','--only-stage','semantics','--output',$outputPath,'--summary','reports/qwen_ground_region_video_only_audit_20260714_summary.csv','--checkpoint-every','1')
foreach ($id in $ids) { $args += @('--sample-id', $id) }
Write-Output "Remaining ground-ball samples: $($ids.Count)"
& python @args
exit $LASTEXITCODE
