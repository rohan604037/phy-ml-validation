"""Configuration loading and output-folder layout."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "test_plan.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "output"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class Paths:
    """Where every artefact is written. Everything lives under one output folder."""
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        for p in (self.data, self.logs, self.models, self.plots, self.reports):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def reports(self) -> Path:
        return self.root / "reports"
