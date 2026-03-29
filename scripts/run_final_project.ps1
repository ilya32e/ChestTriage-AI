$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

New-Item -ItemType Directory -Force -Path ".\artifacts\logs" | Out-Null

Write-Host "[1/10] Generating EDA artifacts..."
python .\scripts\generate_eda.py --root data/medmnist --size 128 --output-dir artifacts/eda

Write-Host "[2/10] Training supervised model: SimpleCNN..."
python .\scripts\train_supervised.py --config .\configs\supervised\simple_cnn.yaml

Write-Host "[3/10] Calibrating supervised thresholds for the deployed SimpleCNN checkpoint..."
python .\scripts\calibrate_supervised_thresholds.py --checkpoint .\artifacts\supervised\simple_cnn\best_model.pt

Write-Host "[4/10] Training supervised model: ResNet18 transfer learning..."
python .\scripts\train_supervised.py --config .\configs\supervised\resnet18_transfer.yaml

Write-Host "[5/10] Training supervised model: TinyViT..."
python .\scripts\train_supervised.py --config .\configs\supervised\tiny_vit.yaml

Write-Host "[6/10] Training anomaly model: Conv Autoencoder..."
python .\scripts\train_anomaly.py --config .\configs\anomaly\conv_autoencoder.yaml

Write-Host "[7/10] Training multimodal baseline: NIH image only..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\image_only_nih_metadata.yaml

Write-Host "[8/10] Training multimodal baseline: NIH metadata only..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\text_only_nih_metadata.yaml

Write-Host "[9/10] Training multimodal fusion model: NIH image + metadata..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\fusion_nih_metadata.yaml

Write-Host "[10/10] Refreshing report tables and deployment manifest..."
python .\scripts\export_report_tables.py

Write-Host "[Final] Starting Streamlit app..."
python -m streamlit run .\app\streamlit_app.py --server.address 127.0.0.1 --server.port 8501
