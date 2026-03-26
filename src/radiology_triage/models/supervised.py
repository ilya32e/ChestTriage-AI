from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_labels: int = 14, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class TinyViT(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_dim: int = 384,
        dropout: float = 0.1,
        num_labels: int = 14,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size for TinyViT.")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_labels)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(inputs)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.pos_dropout(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def build_supervised_model(
    model_name: str,
    num_labels: int,
    pretrained: bool = True,
    dropout: float = 0.2,
    in_channels: int = 3,
    image_size: int = 224,
) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "simple_cnn":
        return SimpleCNN(in_channels=in_channels, num_labels=num_labels, dropout=dropout)
    if model_name == "tiny_vit":
        return TinyViT(
            image_size=image_size,
            patch_size=16,
            in_channels=in_channels,
            embed_dim=192,
            depth=4,
            num_heads=4,
            mlp_dim=384,
            dropout=dropout,
            num_labels=num_labels,
        )
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_labels))
        return model
    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier.in_features, num_labels))
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_labels),
        )
        return model
    if model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_labels)
        return model
    raise ValueError(f"Unsupported supervised model: {model_name}")


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    model_name = model_name.lower()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if model_name == "resnet18":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif model_name == "densenet121":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif model_name == "efficientnet_b0":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif model_name == "vit_b_16":
        for parameter in model.heads.parameters():
            parameter.requires_grad = True
    elif model_name == "simple_cnn":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif model_name == "tiny_vit":
        for parameter in model.head.parameters():
            parameter.requires_grad = True
        model.cls_token.requires_grad = False
        model.pos_embedding.requires_grad = False
    else:
        raise ValueError(f"Unsupported model for freezing: {model_name}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
