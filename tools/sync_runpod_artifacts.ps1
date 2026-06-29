<#
.SYNOPSIS
Download only expensive RunPod artifacts from the RunPod S3 bucket.

.DESCRIPTION
This replaces a full-bucket sync such as:
  aws s3 sync --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io s3://0wov6gbp6j/ C:\runpod_data\

The include list is based on Pod/run.ipynb:
  - text_h5.h5 from the split/text-H5 build stage
  - embedding_h5.h5 from the cloud embedding stage
  - VICReg_review/heads experiment outputs
  - timestamped stable_query_latent_artifacts archives and logs
#>

[CmdletBinding()]
param(
    [string]$Source = "s3://0wov6gbp6j/",
    [string]$Destination = "C:\runpod_data\",
    [string]$Region = "us-ks-2",
    [string]$EndpointUrl = "https://s3api-us-ks-2.runpod.io",
    [string]$AwsCliPath = "aws",
    [switch]$DryRun,
    [switch]$PrintOnly,
    [switch]$ShowCommand,
    [switch]$RequireFingerprintMatch
)

$ErrorActionPreference = "Stop"

# Preferred path: timestamped archives created by the final collection cell in
# Pod/run.ipynb. Fallback paths cover buckets that mirror /workspace directly.
$includePatterns = @(
    "stable_query_latent_artifacts/*",
    "workspace/stable_query_latent_artifacts/*",
    "stable_query_latent_logs/*",
    "workspace/stable_query_latent_logs/*",

    "*embedding_h5.h5",
    "*embedding_h5.h5.cloud_manifest.json",
    "*embedding_h5.h5.incloud_manifest.json",
    "*text_h5.h5",
    "*text_h5.h5.manifest.json",

    "game_review_data/embedding_h5.h5",
    "game_review_data/embedding_h5.h5.cloud_manifest.json",
    "game_review_data/embedding_h5.h5.incloud_manifest.json",
    "game_review_data/build_new_gamedata/text_h5.h5",
    "game_review_data/build_new_gamedata/text_h5.h5.manifest.json",
    "VICReg_review/heads/*",

    "stable-query-latent/game_review_data/embedding_h5.h5",
    "stable-query-latent/game_review_data/embedding_h5.h5.cloud_manifest.json",
    "stable-query-latent/game_review_data/embedding_h5.h5.incloud_manifest.json",
    "stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5",
    "stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5.manifest.json",
    "stable-query-latent/VICReg_review/heads/*",

    "workspace/stable-query-latent/game_review_data/embedding_h5.h5",
    "workspace/stable-query-latent/game_review_data/embedding_h5.h5.cloud_manifest.json",
    "workspace/stable-query-latent/game_review_data/embedding_h5.h5.incloud_manifest.json",
    "workspace/stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5",
    "workspace/stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5.manifest.json",
    "workspace/stable-query-latent/VICReg_review/heads/*"
)

function Format-CommandArg {
    param([string]$Arg)
    if ($Arg -match '[\s"]') {
        return '"' + ($Arg -replace '"', '\"') + '"'
    }
    return $Arg
}

function ConvertFrom-S3Uri {
    param([string]$Uri)

    if ($Uri -notmatch '^s3://([^/]+)(?:/(.*))?$') {
        throw "Source must be an s3:// URI: $Uri"
    }

    $prefix = ""
    if ($Matches[2]) {
        $prefix = $Matches[2].TrimStart("/")
    }
    if ($prefix -and -not $prefix.EndsWith("/")) {
        $prefix += "/"
    }

    [PSCustomObject]@{
        Bucket = $Matches[1]
        Prefix = $prefix
    }
}

function Join-LocalS3Key {
    param(
        [string]$Root,
        [string]$RelativeKey
    )

    $localRelative = $RelativeKey -replace '/', [System.IO.Path]::DirectorySeparatorChar
    Join-Path $Root $localRelative
}

function Normalize-ETag {
    param($ETag)
    if ($null -eq $ETag) {
        return ""
    }
    return ([string]$ETag).Trim('"')
}

function Get-S3MetaPath {
    param([string]$LocalPath)
    return "$LocalPath.s3meta.json"
}

function Write-S3Meta {
    param(
        [string]$LocalPath,
        [string]$Bucket,
        $RemoteObject
    )

    $metadata = [PSCustomObject]@{
        bucket = $Bucket
        key = [string]$RemoteObject.Key
        size = [Int64]$RemoteObject.Size
        etag = Normalize-ETag $RemoteObject.ETag
        last_modified = [string]$RemoteObject.LastModified
        endpoint_url = $EndpointUrl
        downloaded_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    }
    $metaPath = Get-S3MetaPath $LocalPath
    $metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding UTF8
}

function Test-PatternIncluded {
    param([string]$RelativeKey)

    foreach ($pattern in $includePatterns) {
        if ($RelativeKey -like $pattern) {
            return $true
        }
    }
    return $false
}

function Test-LocalMatchesRemote {
    param(
        [string]$LocalPath,
        [string]$Bucket,
        $RemoteObject
    )

    if (-not (Test-Path -LiteralPath $LocalPath -PathType Leaf)) {
        return [PSCustomObject]@{ Matches = $false; Reason = "missing local file" }
    }

    $localItem = Get-Item -LiteralPath $LocalPath
    $remoteSize = [Int64]$RemoteObject.Size
    if ([Int64]$localItem.Length -ne $remoteSize) {
        return [PSCustomObject]@{
            Matches = $false
            Reason = "size differs local=$($localItem.Length) remote=$remoteSize"
        }
    }

    $remoteEtag = Normalize-ETag $RemoteObject.ETag
    $metaPath = Get-S3MetaPath $LocalPath
    if (Test-Path -LiteralPath $metaPath -PathType Leaf) {
        try {
            $meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
            if (
                [string]$meta.bucket -eq $Bucket -and
                [string]$meta.key -eq [string]$RemoteObject.Key -and
                [Int64]$meta.size -eq $remoteSize -and
                (Normalize-ETag $meta.etag) -eq $remoteEtag
            ) {
                return [PSCustomObject]@{ Matches = $true; Reason = "size and saved ETag match" }
            }
        }
        catch {
            Write-Warning "Ignoring unreadable metadata sidecar: $metaPath"
        }
    }

    if ($remoteEtag -match '^[A-Fa-f0-9]{32}$') {
        $localMd5 = (Get-FileHash -LiteralPath $LocalPath -Algorithm MD5).Hash.ToLowerInvariant()
        if ($localMd5 -eq $remoteEtag.ToLowerInvariant()) {
            Write-S3Meta -LocalPath $LocalPath -Bucket $Bucket -RemoteObject $RemoteObject
            return [PSCustomObject]@{ Matches = $true; Reason = "size and MD5 ETag match" }
        }
        return [PSCustomObject]@{ Matches = $false; Reason = "MD5 ETag differs" }
    }

    if ($RequireFingerprintMatch) {
        return [PSCustomObject]@{
            Matches = $false
            Reason = "size matches but no verifiable local fingerprint sidecar"
        }
    }

    return [PSCustomObject]@{
        Matches = $true
        Reason = "size matches; multipart ETag cannot be recomputed locally"
    }
}

function Test-DownloadedSizeMatchesRemote {
    param(
        [string]$LocalPath,
        $RemoteObject
    )

    if (-not (Test-Path -LiteralPath $LocalPath -PathType Leaf)) {
        return [PSCustomObject]@{ Matches = $false; Reason = "missing downloaded file" }
    }

    $localItem = Get-Item -LiteralPath $LocalPath
    $remoteSize = [Int64]$RemoteObject.Size
    if ([Int64]$localItem.Length -ne $remoteSize) {
        return [PSCustomObject]@{
            Matches = $false
            Reason = "size differs local=$($localItem.Length) remote=$remoteSize"
        }
    }

    return [PSCustomObject]@{ Matches = $true; Reason = "size matches" }
}

function Get-RemoteObjects {
    param(
        [string]$Bucket,
        [string]$Prefix
    )

    $continuationToken = $null
    do {
        $listArgs = @(
            "s3api", "list-objects-v2",
            "--region", $Region,
            "--endpoint-url", $EndpointUrl,
            "--bucket", $Bucket,
            "--prefix", $Prefix,
            "--output", "json"
        )
        if ($continuationToken) {
            $listArgs += @("--continuation-token", $continuationToken)
        }

        if ($ShowCommand) {
            Write-Host ($AwsCliPath + " " + (($listArgs | ForEach-Object { Format-CommandArg $_ }) -join " "))
        }

        $json = & $AwsCliPath @listArgs
        if ($LASTEXITCODE -ne 0) {
            throw "AWS CLI list-objects-v2 failed with exit code $LASTEXITCODE"
        }
        if (-not $json) {
            break
        }

        $page = $json | ConvertFrom-Json
        foreach ($object in @($page.Contents)) {
            if ($object.Key -and -not ([string]$object.Key).EndsWith("/")) {
                $object
            }
        }
        $continuationToken = $page.NextContinuationToken
    } while ($continuationToken)
}

$parsedSource = ConvertFrom-S3Uri $Source
$listPreview = @(
    "s3api", "list-objects-v2",
    "--region", $Region,
    "--endpoint-url", $EndpointUrl,
    "--bucket", $parsedSource.Bucket,
    "--prefix", $parsedSource.Prefix
)
$commandPreview = $AwsCliPath + " " + (($listPreview | ForEach-Object { Format-CommandArg $_ }) -join " ")

if ($PrintOnly) {
    Write-Host $commandPreview
    Write-Host "# Matching objects are copied one-by-one after local size/ETag checks."
    return
}

Write-Host "RunPod selective artifact sync"
Write-Host "  Source      : $Source"
Write-Host "  Destination : $Destination"
Write-Host "  Mode        : $(if ($DryRun) { 'dry-run preview' } else { 'download' })"
Write-Host "  Includes    : text_h5.h5, embedding_h5.h5, manifests, VICReg_review/heads, artifacts, logs"
Write-Host "  Resume      : skip when local size and saved/computable fingerprint match"
if (-not $RequireFingerprintMatch) {
    Write-Host "                multipart ETags fall back to size-only when no sidecar exists"
}
if ($ShowCommand) {
    Write-Host ""
    Write-Host $commandPreview
}
Write-Host ""
Write-Host "Scanning S3 and checking local files before downloading..."
Write-Host "Press Ctrl+C to cancel."
Write-Host ""

if (-not (Get-Command $AwsCliPath -ErrorAction SilentlyContinue)) {
    throw "AWS CLI not found: $AwsCliPath"
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$matched = 0
$skipped = 0
$downloaded = 0
$failed = 0

foreach ($remoteObject in Get-RemoteObjects -Bucket $parsedSource.Bucket -Prefix $parsedSource.Prefix) {
    $key = [string]$remoteObject.Key
    $relativeKey = $key
    if ($parsedSource.Prefix -and $key.StartsWith($parsedSource.Prefix)) {
        $relativeKey = $key.Substring($parsedSource.Prefix.Length)
    }

    if (-not (Test-PatternIncluded $relativeKey)) {
        continue
    }

    $matched += 1
    $destinationPath = Join-LocalS3Key -Root $Destination -RelativeKey $relativeKey
    $match = Test-LocalMatchesRemote -LocalPath $destinationPath -Bucket $parsedSource.Bucket -RemoteObject $remoteObject

    if ($match.Matches) {
        $skipped += 1
        Write-Host "[skip] $relativeKey ($($match.Reason))"
        continue
    }

    $sourceUri = "s3://$($parsedSource.Bucket)/$key"
    if ($DryRun) {
        Write-Host "[dryrun] download $relativeKey ($($match.Reason))"
        continue
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destinationPath) | Out-Null
    Write-Host "[download] $relativeKey ($($match.Reason))"
    & $AwsCliPath s3 cp `
        --region $Region `
        --endpoint-url $EndpointUrl `
        $sourceUri `
        $destinationPath

    if ($LASTEXITCODE -ne 0) {
        $failed += 1
        Write-Warning "Download failed: $relativeKey"
        continue
    }

    $afterDownload = Test-DownloadedSizeMatchesRemote -LocalPath $destinationPath -RemoteObject $remoteObject
    if (-not $afterDownload.Matches) {
        $failed += 1
        Write-Warning "Downloaded file did not pass verification: $relativeKey ($($afterDownload.Reason))"
        continue
    }

    Write-S3Meta -LocalPath $destinationPath -Bucket $parsedSource.Bucket -RemoteObject $remoteObject
    $downloaded += 1
}

Write-Host ""
Write-Host "RunPod sync summary"
Write-Host "  matched    : $matched"
Write-Host "  skipped    : $skipped"
Write-Host "  downloaded : $downloaded"
Write-Host "  failed     : $failed"

if ($failed -gt 0) {
    exit 1
}
exit 0
