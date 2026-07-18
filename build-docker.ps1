#!/usr/bin/env pwsh

$Version = "0.0.13"

$ErrorActionPreference = 'Stop'

$ProjectRoot = $PSScriptRoot
$Dockerfile  = Join-Path $ProjectRoot 'docker/Dockerfile'
$Tag         = 'registries.mars.abramad.com/mars/aid/cocoindex-code-lance-hsnw:' + $Version

# BuildKit is required for the `--mount=type=bind,...,rw=true` in the final layer.
$env:DOCKER_BUILDKIT = '1'

Write-Host "Building $Tag from $Dockerfile" -ForegroundColor Cyan

docker build `
    --file $Dockerfile `
    --tag $Tag `
    $ProjectRoot

if ($LASTEXITCODE -ne 0) {
    Write-Error "docker build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Built and tagged $Tag" -ForegroundColor Green
