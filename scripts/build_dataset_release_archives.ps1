param(
    [string]$OutputDirectory = "release_assets_20260714",
    [long]$TargetBytes = 6MB
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
$output = Join-Path $repo $OutputDirectory
if (Test-Path $output) {
    throw "Output directory already exists: $output"
}
New-Item -ItemType Directory -Path $output | Out-Null

$samples = @(
    Get-ChildItem (Join-Path $repo "dataset") -Recurse -Filter sample.csv |
        ForEach-Object {
            $directory = $_.Directory
            $files = @(Get-ChildItem $directory.FullName -File)
            [pscustomobject]@{
                SampleId = (Import-Csv $_.FullName | Select-Object -First 1).sample_id
                Directory = $directory
                Relative = $directory.FullName.Substring($repo.Length + 1).Replace("\", "/")
                Bytes = ($files | Measure-Object Length -Sum).Sum
            }
        } |
        Sort-Object SampleId
)

$groups = @()
$current = @()
$currentBytes = 0L
foreach ($sample in $samples) {
    if ($current.Count -gt 0 -and $currentBytes + $sample.Bytes -gt $TargetBytes) {
        $groups += ,@($current)
        $current = @()
        $currentBytes = 0L
    }
    $current += $sample
    $currentBytes += $sample.Bytes
}
if ($current.Count -gt 0) { $groups += ,@($current) }

$archiveRows = @()
$sampleRows = @()
for ($index = 0; $index -lt $groups.Count; $index++) {
    $archiveName = "baseball_dataset_384_part_{0:D3}.zip" -f ($index + 1)
    $archivePath = Join-Path $output $archiveName
    $stream = [System.IO.File]::Open($archivePath, [System.IO.FileMode]::CreateNew)
    $zip = [System.IO.Compression.ZipArchive]::new($stream, [System.IO.Compression.ZipArchiveMode]::Create, $false)
    try {
        foreach ($sample in $groups[$index]) {
            foreach ($file in Get-ChildItem $sample.Directory.FullName -File) {
                $entry = $file.FullName.Substring($repo.Length + 1).Replace("\", "/")
                [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                    $zip, $file.FullName, $entry, [System.IO.Compression.CompressionLevel]::Optimal
                ) | Out-Null
            }
            $sampleRows += [pscustomobject]@{
                sample_id = $sample.SampleId
                relative_path = $sample.Relative
                archive = $archiveName
            }
        }
    }
    finally {
        $zip.Dispose()
        $stream.Dispose()
    }
    $info = Get-Item $archivePath
    $archiveRows += [pscustomobject]@{
        archive = $archiveName
        bytes = $info.Length
        sha256 = (Get-FileHash $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        sample_count = $groups[$index].Count
    }
    Write-Output "Created $archiveName samples=$($groups[$index].Count) bytes=$($info.Length)"
}

$archiveRows | Export-Csv (Join-Path $output "archives_manifest.csv") -NoTypeInformation -Encoding utf8
$sampleRows | Export-Csv (Join-Path $output "sample_to_archive.csv") -NoTypeInformation -Encoding utf8
Copy-Item (Join-Path $repo "reports\dataset_index_20260713.csv") (Join-Path $output "dataset_index_20260713.csv")
Write-Output "ARCHIVES_READY count=$($archiveRows.Count) samples=$($samples.Count) output=$output"
