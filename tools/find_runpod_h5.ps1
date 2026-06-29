<#
.SYNOPSIS
Find and optionally download RunPod H5 artifacts without scanning the whole bucket.

.DESCRIPTION
This checks exact likely keys used by the Pod notebooks and repo mirror. Use it
when a broad aws s3 sync or recursive aws s3 ls stays quiet for a long time.
#>

[CmdletBinding()]
param(
    [string]$Source = "s3://0wov6gbp6j/",
    [string]$Destination = "C:\runpod_data\",
    [string]$Region = "us-ks-2",
    [string]$EndpointUrl = "https://s3api-us-ks-2.runpod.io",
    [string]$AwsCliPath = "aws",
    [switch]$Download,
    [switch]$RequireFingerprintMatch
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command $AwsCliPath -ErrorAction SilentlyContinue)) {
    throw "AWS CLI not found: $AwsCliPath"
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
        [string]$Key,
        $Head
    )

    $metadata = [PSCustomObject]@{
        bucket = $Bucket
        key = $Key
        size = [Int64]$Head.ContentLength
        etag = Normalize-ETag $Head.ETag
        last_modified = [string]$Head.LastModified
        endpoint_url = $EndpointUrl
        downloaded_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    }
    $metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Get-S3MetaPath $LocalPath) -Encoding UTF8
}

function Test-LocalMatchesRemote {
    param(
        [string]$LocalPath,
        [string]$Bucket,
        [string]$Key,
        $Head
    )

    if (-not (Test-Path -LiteralPath $LocalPath -PathType Leaf)) {
        return [PSCustomObject]@{ Matches = $false; Reason = "missing local file" }
    }

    $localItem = Get-Item -LiteralPath $LocalPath
    $remoteSize = [Int64]$Head.ContentLength
    if ([Int64]$localItem.Length -ne $remoteSize) {
        return [PSCustomObject]@{
            Matches = $false
            Reason = "size differs local=$($localItem.Length) remote=$remoteSize"
        }
    }

    $remoteEtag = Normalize-ETag $Head.ETag
    $metaPath = Get-S3MetaPath $LocalPath
    if (Test-Path -LiteralPath $metaPath -PathType Leaf) {
        try {
            $meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
            if (
                [string]$meta.bucket -eq $Bucket -and
                [string]$meta.key -eq $Key -and
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
            Write-S3Meta -LocalPath $LocalPath -Bucket $Bucket -Key $Key -Head $Head
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
        $Head
    )

    if (-not (Test-Path -LiteralPath $LocalPath -PathType Leaf)) {
        return [PSCustomObject]@{ Matches = $false; Reason = "missing downloaded file" }
    }

    $localItem = Get-Item -LiteralPath $LocalPath
    $remoteSize = [Int64]$Head.ContentLength
    if ([Int64]$localItem.Length -ne $remoteSize) {
        return [PSCustomObject]@{
            Matches = $false
            Reason = "size differs local=$($localItem.Length) remote=$remoteSize"
        }
    }

    return [PSCustomObject]@{ Matches = $true; Reason = "size matches" }
}

$sourceMatch = [regex]::Match($Source.TrimEnd("/") + "/", '^s3://([^/]+)/(.*)$')
if (-not $sourceMatch.Success) {
    throw "Source must be an s3:// URI: $Source"
}
$bucket = $sourceMatch.Groups[1].Value
$sourcePrefix = $sourceMatch.Groups[2].Value

$sourceRoot = $Source.TrimEnd("/") + "/"
$candidateKeys = @(
    "stable-query-latent/game_review_data/embedding_h5.h5",
    "stable-query-latent/game_review_data/embedding_h5.h5.cloud_manifest.json",
    "stable-query-latent/game_review_data/embedding_h5.h5.incloud_manifest.json",
    "stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5",
    "stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5.manifest.json",

    "workspace/stable-query-latent/game_review_data/embedding_h5.h5",
    "workspace/stable-query-latent/game_review_data/embedding_h5.h5.cloud_manifest.json",
    "workspace/stable-query-latent/game_review_data/embedding_h5.h5.incloud_manifest.json",
    "workspace/stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5",
    "workspace/stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5.manifest.json",

    "game_review_data/embedding_h5.h5",
    "game_review_data/embedding_h5.h5.cloud_manifest.json",
    "game_review_data/embedding_h5.h5.incloud_manifest.json",
    "game_review_data/build_new_gamedata/text_h5.h5",
    "game_review_data/build_new_gamedata/text_h5.h5.manifest.json"
)

$matches = New-Object System.Collections.Generic.List[string]

foreach ($key in $candidateKeys) {
    $uri = $sourceRoot + $key
    Write-Host "Checking $key"
    $output = & $AwsCliPath s3 ls `
        --region $Region `
        --endpoint-url $EndpointUrl `
        $uri 2>$null

    if ($LASTEXITCODE -eq 0 -and $output) {
        $matches.Add($key)
        Write-Host "  FOUND $key"
    }
}

$uniqueMatches = $matches | Sort-Object -Unique

if (-not $uniqueMatches) {
    Write-Host ""
    Write-Host "No text_h5.h5 or embedding_h5.h5 files were found at the expected exact keys."
    Write-Host "That usually means the H5 files were not uploaded to this S3 bucket, or they are under an unexpected prefix."
    exit 2
}

Write-Host ""
Write-Host "Matched files:"
$uniqueMatches | ForEach-Object { Write-Host "  $_" }

if (-not $Download) {
    Write-Host ""
    Write-Host "Add -Download to copy these files into $Destination"
    exit 0
}

foreach ($key in $uniqueMatches) {
    $src = $sourceRoot + $key
    $dst = Join-Path $Destination $key
    $remoteKey = $sourcePrefix + $key
    $headJson = & $AwsCliPath s3api head-object `
        --region $Region `
        --endpoint-url $EndpointUrl `
        --bucket $bucket `
        --key $remoteKey `
        --output json

    if ($LASTEXITCODE -ne 0 -or -not $headJson) {
        Write-Warning "Could not read remote metadata, downloading without precheck: $key"
        $head = $null
    }
    else {
        $head = $headJson | ConvertFrom-Json
        $match = Test-LocalMatchesRemote -LocalPath $dst -Bucket $bucket -Key $remoteKey -Head $head
        if ($match.Matches) {
            Write-Host "Skipping $key ($($match.Reason))"
            continue
        }
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    if ($head) {
        Write-Host "Downloading $key ($($match.Reason))"
    }
    else {
        Write-Host "Downloading $key"
    }
    & $AwsCliPath s3 cp `
        --region $Region `
        --endpoint-url $EndpointUrl `
        $src `
        $dst

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    if ($head) {
        $afterDownload = Test-DownloadedSizeMatchesRemote -LocalPath $dst -Head $head
        if (-not $afterDownload.Matches) {
            throw "Downloaded file did not pass verification: $key ($($afterDownload.Reason))"
        }
        Write-S3Meta -LocalPath $dst -Bucket $bucket -Key $remoteKey -Head $head
    }
}
