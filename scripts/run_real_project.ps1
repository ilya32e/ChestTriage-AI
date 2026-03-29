$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

New-Item -ItemType Directory -Force -Path ".\artifacts\logs" | Out-Null

Write-Host "[1/6] Training supervised runtime model (SimpleCNN on ChestMNIST)..."
python .\scripts\train_supervised.py --config .\configs\supervised\simple_cnn.yaml

Write-Host "[2/6] Calibrating supervised thresholds for the deployed checkpoint..."
python .\scripts\calibrate_supervised_thresholds.py --checkpoint .\artifacts\supervised\simple_cnn\best_model.pt

Write-Host "[3/6] Training anomaly runtime model (Conv Autoencoder on ChestMNIST)..."
python .\scripts\train_anomaly.py --config .\configs\anomaly\conv_autoencoder.yaml

Write-Host "[4/6] Training multimodal runtime model (NIH image only)..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\image_only_nih_metadata.yaml

Write-Host "[5/6] Training multimodal runtime model (NIH metadata only)..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\text_only_nih_metadata.yaml

Write-Host "[6/6] Training multimodal runtime model (NIH image + metadata fusion)..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\fusion_nih_metadata.yaml

Write-Host "[Final] Refreshing deployment manifest and starting Streamlit app..."
python .\scripts\export_report_tables.py
python -m streamlit run .\app\streamlit_app.py --server.address 127.0.0.1 --server.port 8501
