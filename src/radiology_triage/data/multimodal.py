from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def simple_tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


class Vocabulary:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {index: token for token, index in stoi.items()}

    @classmethod
    def build(
        cls,
        texts: list[str],
        min_freq: int = 2,
        max_size: int = 20000,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))
        stoi = {cls.PAD_TOKEN: 0, cls.UNK_TOKEN: 1}
        for token, frequency in counter.most_common():
            if frequency < min_freq or len(stoi) >= max_size:
                continue
            stoi[token] = len(stoi)
        return cls(stoi)

    def encode(self, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = simple_tokenize(text)[:max_length]
        token_ids = [self.stoi.get(token, self.stoi[self.UNK_TOKEN]) for token in tokens]
        attention_mask = [1] * len(token_ids)
        while len(token_ids) < max_length:
            token_ids.append(self.stoi[self.PAD_TOKEN])
            attention_mask.append(0)
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.float32),
        )

    def state_dict(self) -> dict[str, Any]:
        return {"stoi": self.stoi}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Vocabulary":
        return cls(stoi=state["stoi"])

    @property
    def size(self) -> int:
        return len(self.stoi)


class MultimodalRadiologyDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_columns: list[str],
        vocab: Vocabulary,
        max_length: int,
        image_transform: transforms.Compose,
        image_column: str,
        text_column: str,
        base_dir: Path,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.label_columns = label_columns
        self.vocab = vocab
        self.max_length = max_length
        self.image_transform = image_transform
        self.image_column = image_column
        self.text_column = text_column
        self.base_dir = base_dir

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        image_path = (self.base_dir / str(row[self.image_column])).resolve()
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)
        input_ids, attention_mask = self.vocab.encode(str(row.get(self.text_column, "")), self.max_length)
        labels = torch.tensor(row[self.label_columns].values.astype("float32"))
        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "report_text": str(row.get(self.text_column, "")),
            "image_path": str(image_path),
        }


@dataclass
class MultimodalLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    vocab: Vocabulary
    label_names: list[str]


def build_multimodal_loaders(config: dict) -> MultimodalLoaders:
    dataset_cfg = config["dataset"]
    csv_path = Path(dataset_cfg["csv_path"])
    dataframe = pd.read_csv(csv_path)

    split_column = dataset_cfg.get("split_column", "split")
    image_column = dataset_cfg.get("image_column", "image_path")
    text_column = dataset_cfg.get("text_column", "report_text")
    label_columns = dataset_cfg["label_columns"]
    base_dir = Path(dataset_cfg.get("image_base_dir", csv_path.parent))

    seed = config.get("seed", 42)
    train_df = dataframe[dataframe[split_column] == "train"].copy()
    val_df = dataframe[dataframe[split_column] == "val"].copy()
    test_df = dataframe[dataframe[split_column] == "test"].copy()
    train_df = _maybe_limit_dataframe(train_df, dataset_cfg.get("max_train_samples"), seed)
    val_df = _maybe_limit_dataframe(val_df, dataset_cfg.get("max_val_samples"), seed + 1)
    test_df = _maybe_limit_dataframe(test_df, dataset_cfg.get("max_test_samples"), seed + 2)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("The multimodal CSV must contain train, val and test splits.")

    vocab = Vocabulary.build(
        texts=train_df[text_column].fillna("").astype(str).tolist(),
        min_freq=dataset_cfg.get("min_token_freq", 2),
        max_size=dataset_cfg.get("max_vocab_size", 20000),
    )
    image_size = dataset_cfg["image_size"]
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    loader_kwargs = {
        "batch_size": dataset_cfg["batch_size"],
        "num_workers": dataset_cfg.get("num_workers", 0),
        "pin_memory": bool(torch.cuda.is_available()),
    }
    return MultimodalLoaders(
        train=DataLoader(
            MultimodalRadiologyDataset(
                dataframe=train_df,
                label_columns=label_columns,
                vocab=vocab,
                max_length=dataset_cfg["max_length"],
                image_transform=train_transform,
                image_column=image_column,
                text_column=text_column,
                base_dir=base_dir,
            ),
            shuffle=True,
            **loader_kwargs,
        ),
        val=DataLoader(
            MultimodalRadiologyDataset(
                dataframe=val_df,
                label_columns=label_columns,
                vocab=vocab,
                max_length=dataset_cfg["max_length"],
                image_transform=eval_transform,
                image_column=image_column,
                text_column=text_column,
                base_dir=base_dir,
            ),
            shuffle=False,
            **loader_kwargs,
        ),
        test=DataLoader(
            MultimodalRadiologyDataset(
                dataframe=test_df,
                label_columns=label_columns,
                vocab=vocab,
                max_length=dataset_cfg["max_length"],
                image_transform=eval_transform,
                image_column=image_column,
                text_column=text_column,
                base_dir=base_dir,
            ),
            shuffle=False,
            **loader_kwargs,
        ),
        vocab=vocab,
        label_names=label_columns,
    )


def _maybe_limit_dataframe(dataframe: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or len(dataframe) <= max_samples:
        return dataframe
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer.")
    return dataframe.sample(n=max_samples, random_state=seed).reset_index(drop=True)
