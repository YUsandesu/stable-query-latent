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
    [switch]$PrintOnly
)

$ErrorActionPreference = "Stop"

# Preferred path: timestamped archives created by the final collection cell in
# Pod/run.ipynb. Fallback paths cover buckets that mirror /workspace directly.
$includePatterns = @(
    "stable_query_latent_artifacts/*",
    "workspace/stable_query_latent_artifacts/*",
    "stable_query_latent_logs/*",
    "workspace/stable_query_latent_logs/*",

    "game_review_data/embedding_h5.h5",
    "game_review_data/embedding_h5.h5.incloud_manifest.json",
    "game_review_data/build_new_gamedata/text_h5.h5",
    "game_review_data/build_new_gamedata/text_h5.h5.manifest.json",
    "VICReg_review/heads/*",

    "stable-query-latent/game_review_data/embedding_h5.h5",
    "stable-query-latent/game_review_data/embedding_h5.h5.incloud_manifest.json",
    "stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5",
    "stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5.manifest.json",
    "stable-query-latent/VICReg_review/heads/*",

    "workspace/stable-query-latent/game_review_data/embedding_h5.h5",
    "workspace/stable-query-latent/game_review_data/embedding_h5.h5.incloud_manifest.json",
    "workspace/stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5",
    "workspace/stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5.manifest.json",
    "workspace/stable-query-latent/VICReg_review/heads/*"
)

$awsArgs = @(
    "s3", "sync",
    "--region", $Region,
    "--endpoint-url", $EndpointUrl,
    $Source,
    $Destination,
    "--exclude", "*"
)

foreach ($pattern in $includePatterns) {
    $awsArgs += @("--include", $pattern)
}

if ($DryRun) {
    $awsArgs += "--dryrun"
}

function Format-CommandArg {
    param([string]$Arg)
    if ($Arg -match '[\s"]') {
        return '"' + ($Arg -replace '"', '\"') + '"'
    }
    return $Arg
}

$commandPreview = $AwsCliPath + " " + (($awsArgs | ForEach-Object { Format-CommandArg $_ }) -join " ")
Write-Host $commandPreview

if ($PrintOnly) {
    return
}

if (-not (Get-Command $AwsCliPath -ErrorAction SilentlyContinue)) {
    throw "AWS CLI not found: $AwsCliPath"
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
& $AwsCliPath @awsArgs
exit $LASTEXITCODE
