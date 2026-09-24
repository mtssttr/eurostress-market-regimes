"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SeriesConfig:
    name: str
    provider: str
    role: str
    description: str
    unit: str
    redistribution: str
    transformations: tuple[str, ...]
    flow: str | None = None
    key: str | None = None
    symbol: str | None = None


@dataclass(frozen=True)
class ProjectConfig:
    project: dict[str, Any]
    splits: dict[str, str]
    series: tuple[SeriesConfig, ...]
    evaluation: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path = "config/series.yml") -> ProjectConfig:
    """Load the project configuration and reject incomplete series definitions."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise TypeError("Configuration root must be a mapping")

    missing_sections = {"project", "splits", "series"} - raw.keys()
    if missing_sections:
        raise ValueError(f"Missing configuration sections: {sorted(missing_sections)}")

    configured_series: list[SeriesConfig] = []
    for name, values in raw["series"].items():
        provider = values["provider"]
        if provider == "ecb" and not {"flow", "key"} <= values.keys():
            raise ValueError(f"ECB series {name!r} requires flow and key")
        if provider == "yahoo" and "symbol" not in values:
            raise ValueError(f"Yahoo series {name!r} requires a symbol")

        configured_series.append(
            SeriesConfig(
                name=name,
                provider=provider,
                role=values["role"],
                description=values["description"],
                unit=values["unit"],
                redistribution=values["redistribution"],
                transformations=tuple(values.get("transformations", [])),
                flow=values.get("flow"),
                key=values.get("key"),
                symbol=values.get("symbol"),
            )
        )

    return ProjectConfig(
        project=dict(raw["project"]),
        splits=dict(raw["splits"]),
        series=tuple(configured_series),
        evaluation=dict(raw.get("evaluation", {})),
        models=dict(raw.get("models", {})),
    )
