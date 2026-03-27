$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

New-Item -ItemType Directory -Force -Path ".\artifacts\logs" | Out-Null

Write-Host "[1/10] Generating EDA artifacts (224)..."
python .\scripts\generate_eda.py --root data/medmnist --size 224 --output-dir artifacts/eda

Write-Host "[2/10] Training supervised model: SimpleCNN 224..."
python .\scripts\train_supervised.py --config .\configs\supervised\simple_cnn_224.yaml

Write-Host "[3/10] Calibrating supervised thresholds for the deployed SimpleCNN 224 checkpoint..."
python .\scripts\calibrate_supervised_thresholds.py --checkpoint .\artifacts\supervised\simple_cnn_224\best_model.pt

Write-Host "[4/10] Training supervised model: ResNet18 transfer learning 224..."
python .\scripts\train_supervised.py --config .\configs\supervised\resnet18_transfer_224.yaml

Write-Host "[5/10] Training supervised model: TinyViT 224..."
python .\scripts\train_supervised.py --config .\configs\supervised\tiny_vit_224.yaml

Write-Host "[6/10] Training anomaly model: Conv Autoencoder..."
python .\scripts\train_anomaly.py --config .\configs\anomaly\conv_autoencoder.yaml

Write-Host "[7/10] Training multimodal baseline: image only 224..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\image_only_224.yaml

Write-Host "[8/10] Training multimodal baseline: text only 224..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\text_only_224.yaml

Write-Host "[9/10] Training multimodal fusion model 224..."
python .\scripts\train_multimodal.py --config .\configs\multimodal\fusion_224.yaml

Write-Host "[10/10] Starting Streamlit app..."
python -m streamlit run .\app\streamlit_app.py --server.address 127.0.0.1 --server.port 8501
