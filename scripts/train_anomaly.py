from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from radiology_triage.config import load_yaml  # noqa: E402
from radiology_triage.training.anomaly import run_anomaly_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an anomaly detector on ChestMNIST.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    results = run_anomaly_experiment(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

