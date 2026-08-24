"""Canonical repository paths shared by analysis and training entry points."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
EDA_DIR = OUTPUT_DIR / "eda"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
METADATA_DIR = OUTPUT_DIR / "metadata"
METRICS_DIR = OUTPUT_DIR / "metrics"


def ensure_output_directories() -> None:
    """Create the stable generated-artifact directories used by the pipeline."""

    for directory in [EDA_DIR, FIGURES_DIR, LOGS_DIR, METADATA_DIR, METRICS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
