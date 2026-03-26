from __future__ import annotations

import torch
import torch.nn as nn


class SmallImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embedding_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(image))


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        encoded, _ = self.encoder(embedded)
        mask = attention_mask.unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return self.projection(pooled)


class ImageOnlyClassifier(nn.Module):
    def __init__(self, num_labels: int, image_embedding_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.image_encoder = SmallImageEncoder(embedding_dim=image_embedding_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(image_embedding_dim, num_labels),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.image_encoder(image))


class TextOnlyClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        text_embedding_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            hidden_dim=text_embedding_dim,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(text_embedding_dim, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.text_encoder(input_ids, attention_mask))


class FusionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        image_embedding_dim: int = 128,
        text_embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        modality_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_encoder = SmallImageEncoder(embedding_dim=image_embedding_dim)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            hidden_dim=text_embedding_dim,
            dropout=dropout,
        )
        self.modality_dropout = modality_dropout
        self.head = nn.Sequential(
            nn.Linear(image_embedding_dim + text_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        disable_image: bool = False,
        disable_text: bool = False,
    ) -> torch.Tensor:
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)

        if disable_image:
            image_features = torch.zeros_like(image_features)
        if disable_text:
            text_features = torch.zeros_like(text_features)
        if self.training and self.modality_dropout > 0:
            image_features, text_features = self._apply_modality_dropout(image_features, text_features)

        return self.head(torch.cat([image_features, text_features], dim=1))

    def _apply_modality_dropout(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = image_features.shape[0]
        drop_image = torch.rand(batch_size, device=image_features.device) < (self.modality_dropout / 2)
        drop_text = torch.rand(batch_size, device=text_features.device) < (self.modality_dropout / 2)
        image_features = image_features.clone()
        text_features = text_features.clone()
        image_features[drop_image] = 0
        text_features[drop_text] = 0
        return image_features, text_features


def build_multimodal_model(config: dict, vocab_size: int, num_labels: int) -> nn.Module:
    model_cfg = config["model"]
    mode = model_cfg["mode"].lower()
    if mode in {"image", "image_only"}:
        return ImageOnlyClassifier(
            num_labels=num_labels,
            image_embedding_dim=model_cfg.get("image_embedding_dim", 128),
            dropout=model_cfg.get("dropout", 0.2),
        )
    if mode in {"text", "text_only"}:
        return TextOnlyClassifier(
            vocab_size=vocab_size,
            num_labels=num_labels,
            text_embedding_dim=model_cfg.get("text_embedding_dim", 128),
            dropout=model_cfg.get("dropout", 0.2),
        )
    if mode in {"fusion", "multimodal"}:
        return FusionClassifier(
            vocab_size=vocab_size,
            num_labels=num_labels,
            image_embedding_dim=model_cfg.get("image_embedding_dim", 128),
            text_embedding_dim=model_cfg.get("text_embedding_dim", 128),
            hidden_dim=model_cfg.get("fusion_hidden_dim", 256),
            dropout=model_cfg.get("dropout", 0.2),
            modality_dropout=model_cfg.get("modality_dropout", 0.1),
        )
    raise ValueError(f"Unsupported multimodal mode: {mode}")
