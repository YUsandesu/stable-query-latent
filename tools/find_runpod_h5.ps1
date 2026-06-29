<#
.SYNOPSIS
Find and optionally download RunPod H5 artifacts without scanning the whole bucket.

.DESCRIPTION
This checks only likely prefixes used by the Pod notebooks and repo mirror. Use
it when a broad aws s3 sync stays quiet for a long time.
#>

[CmdletBinding()]
param(
    [string]$Source = "s3://0wov6gbp6j/",
    [string]$Destination = "C:\runpod_data\",
    [string]$Region = "us-ks-2",
    [string]$EndpointUrl = "https://s3api-us-ks-2.runpod.io",
    [string]$AwsCliPath = "aws",
    [switch]$Download
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command $AwsCliPath -ErrorAction SilentlyContinue)) {
    throw "AWS CLI not found: $AwsCliPath"
}

$sourceRoot = $Source.TrimEnd("/") + "/"
$prefixes = @(
    "stable-query-latent/game_review_data/",
    "workspace/stable-query-latent/game_review_data/",
    "game_review_data/",
    "stable_query_latent_artifacts/",
    "workspace/stable_query_latent_artifacts/"
)

$patterns = @(
    "embedding_h5\.h5$",
    "embedding_h5\.h5\.incloud_manifest\.json$",
    "text_h5\.h5$",
    "text_h5\.h5\.manifest\.json$"
)
$matchRegex = "(?i)(" + ($patterns -join "|") + ")"

function Get-KeyFromAwsLsLine {
    param([string]$Line)
    if ($Line -match "^\S+\s+\S+\s+\d+\s+(.+)$") {
        return $Matches[1]
    }
    return $null
}

$matches = New-Object System.Collections.Generic.List[string]

foreach ($prefix in $prefixes) {
    $uri = $sourceRoot + $prefix
    Write-Host "Checking $uri"
    $lines = & $AwsCliPath s3 ls `
        --region $Region `
        --endpoint-url $EndpointUrl `
        $uri `
        --recursive

    foreach ($line in $lines) {
        $key = Get-KeyFromAwsLsLine $line
        if ($key -and $key -match $matchRegex) {
            $matches.Add($key)
            Write-Host "  FOUND $key"
        }
    }
}

$uniqueMatches = $matches | Sort-Object -Unique

if (-not $uniqueMatches) {
    Write-Host ""
    Write-Host "No text_h5.h5 or embedding_h5.h5 files were found under the checked prefixes."
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
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Write-Host "Downloading $key"
    & $AwsCliPath s3 cp `
        --region $Region `
        --endpoint-url $EndpointUrl `
        $src `
        $dst
}
