from __future__ import annotations

from pathlib import Path

import mlflow


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_directory_contents(directory: str | Path) -> None:
    directory = Path(directory)
    if not directory.exists():
        return
    for path in directory.rglob("*"):
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path=str(path.parent.relative_to(directory)))
