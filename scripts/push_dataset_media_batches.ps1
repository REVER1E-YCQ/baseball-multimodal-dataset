param(
    [string]$Branch = "codex/high-confidence-qc-20260713",
    [int]$BatchSize = 20,
    [int]$MaxRetries = 3
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo

function Get-RemainingMedia {
    $remotePaths = @(git ls-tree -r --name-only "origin/$Branch" -- dataset)
    if ($LASTEXITCODE -ne 0) { throw "Unable to read origin/$Branch" }
    $remote = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($path in $remotePaths) { [void]$remote.Add($path) }
    return @(
        Get-ChildItem dataset -Recurse -File -Include *.mp4,*.wav |
            ForEach-Object {
                $relativeToDataset = $_.FullName.Substring($_.FullName.IndexOf("\dataset\") + 1)
                $relativeToDataset.Replace("\", "/")
            } |
            Where-Object { -not $remote.Contains($_) } |
            Sort-Object
    )
}

function RemoteContainsBatch([string[]]$Batch) {
    $remotePaths = @(git ls-tree -r --name-only "origin/$Branch" -- dataset)
    if ($LASTEXITCODE -ne 0) { return $false }
    $remote = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($path in $remotePaths) { [void]$remote.Add($path) }
    return @($Batch | Where-Object { -not $remote.Contains($_) }).Count -eq 0
}

$batchNumber = 1
while ($true) {
    git fetch origin $Branch --quiet
    if ($LASTEXITCODE -ne 0) { throw "Unable to fetch origin/$Branch" }
    git reset --mixed "origin/$Branch"
    if ($LASTEXITCODE -ne 0) { throw "Unable to align local branch with origin/$Branch" }
    $remaining = @(Get-RemainingMedia)
    if ($remaining.Count -eq 0) {
        Write-Output "UPLOAD_COMPLETE"
        exit 0
    }
    $batch = @($remaining | Select-Object -First $BatchSize)
    $uploaded = $false
    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        Write-Output "BATCH=$batchNumber ATTEMPT=$attempt REMAINING=$($remaining.Count) FILES=$($batch.Count)"
        git add -f -- $batch
        if ($LASTEXITCODE -ne 0) { throw "git add failed" }
        git -c gc.auto=0 commit -m ("Upload restored media microbatch {0:D3}" -f $batchNumber)
        if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
        git push origin $Branch
        if ($LASTEXITCODE -eq 0) {
            $uploaded = $true
            break
        }
        git fetch origin $Branch --quiet
        if ($LASTEXITCODE -eq 0 -and (RemoteContainsBatch $batch)) {
            Write-Output "BATCH=$batchNumber accepted by remote despite local push error"
            git reset --mixed "origin/$Branch"
            $uploaded = $true
            break
        }
        git reset --mixed "origin/$Branch"
        Start-Sleep -Seconds 10
    }
    if (-not $uploaded) {
        throw "Batch $batchNumber failed after $MaxRetries attempts"
    }
    $batchNumber++
}
