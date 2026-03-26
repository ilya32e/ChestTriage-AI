$ErrorActionPreference = "Stop"

python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python - <<'PY'
import torch
print("torch_version=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
print("torch_cuda_version=", torch.version.cuda)
if torch.cuda.is_available():
    print("device_name=", torch.cuda.get_device_name(0))
PY
