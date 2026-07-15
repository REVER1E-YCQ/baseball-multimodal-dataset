param(
    [string]$AssetsDirectory = "release_assets_20260714",
    [string]$Owner = "REVER1E-YCQ",
    [string]$Repository = "baseball-multimodal-dataset",
    [string]$Tag = "dataset-384-v2026.07.14",
    [int]$MaxRetries = 4
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot
$assetsRoot = Resolve-Path $AssetsDirectory

function Get-Headers {
    $credential = "protocol=https`nhost=github.com`n`n" | git credential fill
    $tokenLine = $credential -split "`n" | Where-Object { $_ -like "password=*" } | Select-Object -First 1
    if (-not $tokenLine) { throw "GitHub credential unavailable" }
    return @{
        Authorization = "Bearer $($tokenLine.Substring(9))"
        Accept = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28"
    }
}

function Get-ReleaseAssets([hashtable]$Headers, [long]$ReleaseId) {
    $result = [System.Collections.Generic.List[object]]::new()
    $page = 1
    while ($true) {
        $items = Invoke-RestMethod -Headers $Headers -Uri "https://api.github.com/repos/$Owner/$Repository/releases/$ReleaseId/assets?per_page=100&page=$page"
        foreach ($item in $items) { $result.Add($item) }
        if ($items.Count -lt 100) { break }
        $page++
    }
    return $result.ToArray()
}

$headers = Get-Headers
$release = $null
try {
    $release = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/$Owner/$Repository/releases/tags/$Tag"
}
catch {
    if ($_.Exception.Response.StatusCode.value__ -ne 404) { throw }
    $release = @(Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/$Owner/$Repository/releases?per_page=100") |
        Where-Object { $_.tag_name -eq $Tag } |
        Select-Object -First 1
}
if (-not $release) {
    $body = @{
        tag_name = $Tag
        target_commitish = "codex/high-confidence-qc-20260713"
        name = "Baseball multimodal dataset — 384 samples"
        draft = $true
        prerelease = $false
        body = "Complete 384-sample dataset release. Download all ZIP parts, then use archives_manifest.csv (SHA256) and sample_to_archive.csv to verify and reconstruct the original dataset/ directory tree."
    } | ConvertTo-Json
    $release = Invoke-RestMethod -Method Post -Headers $headers -ContentType "application/json" -Body $body -Uri "https://api.github.com/repos/$Owner/$Repository/releases"
    Write-Output "Created draft release id=$($release.id)"
}

$assets = @(Get-ChildItem $assetsRoot -File | Sort-Object Name)
$uploadBase = $release.upload_url -replace "\{.*$", ""
foreach ($file in $assets) {
    $remoteAssets = @(Get-ReleaseAssets $headers $release.id)
    $existing = $remoteAssets | Where-Object { $_.name -eq $file.Name } | Select-Object -First 1
    if ($existing -and $existing.size -eq $file.Length) {
        Write-Output "Already uploaded $($file.Name)"
        continue
    }
    if ($existing) {
        Invoke-RestMethod -Method Delete -Headers $headers -Uri "https://api.github.com/repos/$Owner/$Repository/releases/assets/$($existing.id)"
    }
    $uri = "${uploadBase}?name=$([uri]::EscapeDataString($file.Name))"
    $uploaded = $false
    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        try {
            Invoke-WebRequest -Method Post -Headers $headers -ContentType "application/octet-stream" -InFile $file.FullName -Uri $uri | Out-Null
            Write-Output "Uploaded $($file.Name) bytes=$($file.Length)"
            $uploaded = $true
            break
        }
        catch {
            Write-Output "Retry $attempt/$MaxRetries $($file.Name): $($_.Exception.Message)"
            $remoteNow = @(Get-ReleaseAssets $headers $release.id)
            $accepted = $remoteNow | Where-Object { $_.name -eq $file.Name -and $_.size -eq $file.Length } | Select-Object -First 1
            if ($accepted) {
                Write-Output "Accepted by GitHub despite client response error: $($file.Name)"
                $uploaded = $true
                break
            }
            Start-Sleep -Seconds (10 * $attempt)
        }
    }
    if (-not $uploaded) { throw "Failed to upload $($file.Name)" }
}

$remoteAssets = @(Get-ReleaseAssets $headers $release.id)
$missing = @()
foreach ($file in $assets) {
    $match = $remoteAssets | Where-Object { $_.name -eq $file.Name -and $_.size -eq $file.Length } | Select-Object -First 1
    if (-not $match) { $missing += $file }
}
if ($missing.Count -gt 0) { throw "Release verification failed; missing assets: $($missing.Name -join ', ')" }

$publishBody = @{ draft = $false } | ConvertTo-Json
$release = Invoke-RestMethod -Method Patch -Headers $headers -ContentType "application/json" -Body $publishBody -Uri "https://api.github.com/repos/$Owner/$Repository/releases/$($release.id)"
Write-Output "RELEASE_PUBLISHED url=$($release.html_url) assets=$($remoteAssets.Count)"
