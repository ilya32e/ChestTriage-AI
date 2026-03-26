from __future__ import annotations

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_channels: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(inputs)
        return self.decoder(latent)


def build_reconstruction_model(
    model_name: str,
    in_channels: int = 1,
    latent_channels: int = 128,
) -> nn.Module:
    normalized_name = model_name.lower()
    if normalized_name in {"conv_autoencoder", "autoencoder", "ae"}:
        return ConvAutoencoder(in_channels=in_channels, latent_channels=latent_channels)
    if normalized_name in {"conv_vae", "vae"}:
        raise NotImplementedError("A convolutional VAE can be added on top of this builder without changing the API.")
    raise ValueError(f"Unsupported reconstruction model: {model_name}")
