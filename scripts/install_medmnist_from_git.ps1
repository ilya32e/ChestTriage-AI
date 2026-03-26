$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "Installing MedMNIST from the official GitHub repository..."
python -m pip install --upgrade git+https://github.com/MedMNIST/MedMNIST.git

Write-Host "Installed version:"
@'
import medmnist
print(medmnist.__version__)
'@ | python -
