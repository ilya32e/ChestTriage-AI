from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from radiology_triage.models.autoencoder import ConvAutoencoder  # noqa: E402
from radiology_triage.models.multimodal import FusionClassifier, ImageOnlyClassifier, TextOnlyClassifier  # noqa: E402
from radiology_triage.models.supervised import build_supervised_model  # noqa: E402


def main() -> None:
    batch_rgb = torch.randn(2, 3, 224, 224)
    batch_gray = torch.rand(2, 1, 64, 64)
    input_ids = torch.randint(0, 100, (2, 32))
    attention_mask = torch.ones(2, 32)

    simple = build_supervised_model("simple_cnn", num_labels=14, pretrained=False)
    resnet = build_supervised_model("resnet18", num_labels=14, pretrained=False)
    vit = build_supervised_model("tiny_vit", num_labels=14, pretrained=False, image_size=128)
    autoencoder = ConvAutoencoder(in_channels=1)
    image_only = ImageOnlyClassifier(num_labels=3)
    text_only = TextOnlyClassifier(vocab_size=100, num_labels=3)
    fusion = FusionClassifier(vocab_size=100, num_labels=3)

    assert simple(batch_rgb).shape == (2, 14)
    assert resnet(batch_rgb).shape == (2, 14)
    assert vit(torch.randn(2, 3, 128, 128)).shape == (2, 14)
    assert autoencoder(batch_gray).shape == (2, 1, 64, 64)
    assert image_only(batch_rgb).shape == (2, 3)
    assert text_only(input_ids, attention_mask).shape == (2, 3)
    assert fusion(batch_rgb, input_ids, attention_mask).shape == (2, 3)

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
