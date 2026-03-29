$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "[1/4] Training multimodal runtime model (NIH image only, accelerated)..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\image_only_nih_metadata.yaml

Write-Host "[2/4] Training multimodal runtime model (NIH metadata only, accelerated)..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\text_only_nih_metadata.yaml

Write-Host "[3/4] Training multimodal runtime model (NIH image + metadata fusion, accelerated)..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\fusion_nih_metadata.yaml

Write-Host "[4/4] Refreshing deployment manifest..."
python .\scripts\export_report_tables.py
