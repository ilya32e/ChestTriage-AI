from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data)!r}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def flatten_dict(
    data: dict[str, Any],
    prefix: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key, value in data.items():
        composed_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, composed_key, separator))
        elif isinstance(value, (list, tuple, set)):
            items[composed_key] = ",".join(map(str, value))
        elif isinstance(value, Path):
            items[composed_key] = str(value)
        else:
            items[composed_key] = value
    return items


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if base_dir is not None:
        return (Path(base_dir) / candidate).resolve()
    return candidate.resolve()
