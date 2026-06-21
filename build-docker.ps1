#!/usr/bin/env pwsh
# Builds docker/Dockerfile and tags the image cocoindex-code-lance-hsnw:0.0.1.
#
# Run from anywhere — the script resolves paths relative to its own location so
# the build context is the project root (the Dockerfile COPYs docker/entrypoint.sh
# and bind-mounts the whole tree, so the context must be the repo root, not docker/).

$ErrorActionPreference = 'Stop'

$ProjectRoot = $PSScriptRoot
$Dockerfile  = Join-Path $ProjectRoot 'docker/Dockerfile'
$Tag         = 'registries.mars.abramad.com/mars/aid/cocoindex-code-lance-hsnw:0.0.2'

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
