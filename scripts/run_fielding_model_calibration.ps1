$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo
if (-not $env:QWEN_API_KEY) { throw 'QWEN_API_KEY must be set in the launching process.' }

$config = Get-Content 'config/qwen_models.json' -Raw | ConvertFrom-Json
$models = @($config.review_fallback_models + $config.review_realtime_models | Select-Object -Unique)
foreach ($model in $models) {
    $slug = $model -replace '[^A-Za-z0-9_.-]', '_'
    $output = "reports/qwen_fielding_model_calibration_${slug}_20260714.jsonl"
    $summary = "reports/qwen_fielding_model_calibration_${slug}_20260714.csv"
    $env:QWEN_REVIEW_MODEL_FALLBACKS = $model
    # The provider rejects a model once its enabled free quota is exhausted;
    # do not impose a fabricated token ceiling across models with different
    # allowance sizes.
    $env:QWEN_MODEL_TOKEN_CAP = '0'
    Remove-Item -LiteralPath $output, $summary -Force -ErrorAction SilentlyContinue
    Write-Output "Calibrating $model"
    & python scripts/qwen_review_dataset.py --only-stage fielding --sample-id G_153 --sample-id G_192 --output $output --summary $summary --checkpoint-every 1
}
