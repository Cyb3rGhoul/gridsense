from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


DATASETS = {
    "sgcc": {
        "repo": "https://github.com/henryRDlab/ElectricityTheftDetection.git",
        "note": "State Grid theft dataset. Large split zip files; unzip data.zip with data.z01/data.z02 beside it.",
    },
    "lead": {
        "repo": "https://github.com/samy101/lead-dataset.git",
        "note": "LEAD anomaly dataset metadata and available public files. Full competition data may require Kaggle access.",
    },
}


def clone(name: str, url: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / name
    if target.exists():
        print(f"{name}: already exists at {target}")
        return
    subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
    print(f"{name}: cloned to {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public GridSense reference datasets.")
    parser.add_argument("--dataset", choices=[*DATASETS.keys(), "all"], default="all")
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for name, meta in selected.items():
        print(f"\n{name}: {meta['note']}")
        clone(name, meta["repo"])
    print("\nKaggle datasets require a Kaggle token and rule acceptance; keep them optional for the demo.")


if __name__ == "__main__":
    main()
