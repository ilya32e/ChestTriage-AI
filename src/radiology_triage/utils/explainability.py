from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


def resolve_gradcam_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module | None:
    normalized_name = model_name.lower()
    # The demo only exposes Grad-CAM for convolutional backbones with a stable
    # spatial feature map. Transformer variants are intentionally excluded here
    # to avoid a heavier and less reliable explainability path in the UI.
    if normalized_name == "simple_cnn":
        return model.features[12]
    if normalized_name == "resnet18":
        return model.layer4[-1].conv2
    return None


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    target_index: int,
) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output.detach())

    def backward_hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        target_score = logits[:, target_index].sum()
        target_score.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        feature_maps = activations[-1]
        grad_maps = gradients[-1]
        weights = grad_maps.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_maps).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy().astype(np.float32)
    finally:
        forward_handle.remove()
        backward_handle.remove()


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "inferno",
) -> Image.Image:
    base = image.convert("RGB")
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255), mode="L").resize(base.size)
    color_map = colormaps[colormap_name]
    colored = (color_map(np.asarray(heatmap_image, dtype=np.float32) / 255.0)[..., :3] * 255).astype(np.uint8)
    overlay = Image.fromarray(colored, mode="RGB")
    return Image.blend(base, overlay, alpha=alpha)


def build_gradcam_package(
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    target_index: int,
    target_label: str,
) -> dict[str, Any] | None:
    target_layer = resolve_gradcam_target_layer(model, model_name)
    if target_layer is None:
        return None
    heatmap = generate_gradcam(model, input_tensor, target_layer, target_index)
    overlay = overlay_heatmap_on_image(original_image, heatmap)
    return {
        "target_label": target_label,
        "heatmap": heatmap,
        "overlay": overlay,
    }
