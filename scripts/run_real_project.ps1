$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

New-Item -ItemType Directory -Force -Path ".\artifacts\logs" | Out-Null

Write-Host "[1/4] Training supervised runtime model (SimpleCNN on ChestMNIST)..."
python .\scripts\train_supervised.py --config .\configs\supervised\simple_cnn.yaml

Write-Host "[2/4] Training anomaly runtime model (Conv Autoencoder on ChestMNIST)..."
python .\scripts\train_anomaly.py --config .\configs\anomaly\conv_autoencoder.yaml

Write-Host "[3/4] Training multimodal runtime model (IU X-Ray image + text fusion)..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\fusion.yaml

Write-Host "[4/4] Starting Streamlit app..."
python -m streamlit run .\app\streamlit_app.py --server.address 127.0.0.1 --server.port 8501
