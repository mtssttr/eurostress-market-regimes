"""Blocking input controls and machine-readable investigation records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eurostress.audit import summarize_weekly_coverage
from eurostress.config import ProjectConfig


class DataQualityError(ValueError):
    """A calculation was blocked before using a material input exception."""


@dataclass(frozen=True)
class Finding:
    series: str
    rule: str
    severity: str
    affected_rows: int
    explanation: str


def inspect_frame(
    frame: pd.DataFrame,
    *,
    series: str,
    as_of: str,
    expected_unit: str,
    declared_unit: str | None,
    max_age_days: int = 5,
) -> list[Finding]:
    """Validate declared units and observations; never guess units from magnitudes.

    Freshness is measured at the historical run's configured as-of date, not today.
    An absent unit declaration is a warning, not a claim of verified units.
    """
    findings: list[Finding] = []

    def add(rule: str, count: int, explanation: str, severity: str = "ERROR") -> None:
        findings.append(Finding(series, rule, severity, count, explanation))

    if not {"date", "value"} <= set(frame.columns) or frame.empty:
        add("schema", len(frame), "Nonempty date/value columns are required.")
        return findings
    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_localize(None)
    values = pd.to_numeric(frame["value"], errors="coerce")
    invalid_dates = int(dates.isna().sum())
    invalid_values = int((~np.isfinite(values.to_numpy(dtype=float))).sum())
    if invalid_dates:
        add("invalid_date", invalid_dates, "Invalid dates require source investigation.")
    if invalid_values:
        add("missing_or_nonfinite", invalid_values, "Missing/nonfinite values cannot enter models.")
    duplicates = int(dates.duplicated(keep=False).sum())
    if duplicates:
        add("duplicate_date", duplicates, "Resolve duplicate dates against the source.")
    cutoff = pd.Timestamp(as_of)
    future = int((dates > cutoff).sum())
    if future:
        add("future_observation", future, "Observations exceed the declared run date.")
    if dates.notna().any():
        age = int((cutoff - dates.max()).days)
        if age > max_age_days:
            add("stale_feed", len(frame), f"Latest observation is {age} calendar days old.")
    if declared_unit is None:
        add("unit_unverified", len(frame), "Legacy input has no declared source unit.", "WARNING")
    elif declared_unit != expected_unit:
        add("unit_mismatch", len(frame), f"Declared {declared_unit}; expected {expected_unit}.")
    if expected_unit in {"index_points", "usd_per_eur"} and (values <= 0).any():
        add("nonpositive_price", int((values <= 0).sum()), "Prices and FX rates must be positive.")
    return findings


def write_findings(findings: list[Finding], path: str | Path) -> dict:
    payload = {
        "status": "BLOCKED" if any(x.severity == "ERROR" for x in findings) else "PASS",
        "errors": sum(x.severity == "ERROR" for x in findings),
        "warnings": sum(x.severity == "WARNING" for x in findings),
        "findings": [asdict(x) for x in findings],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def enforce_raw_controls(
    config: ProjectConfig,
    directory: str | Path,
    report_path: str | Path = "reports/controls/data_quality.json",
) -> dict:
    """Check every configured source and its stored hash before downstream work."""
    root = Path(directory)
    findings: list[Finding] = []
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {entry["series_name"]: entry for entry in manifest["series"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        findings.append(Finding("all", "manifest", "ERROR", 0, str(exc)))
        entries = {}
    for source in config.series:
        path = root / f"{source.name}.parquet"
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            findings.append(Finding(source.name, "read_input", "ERROR", 0, str(exc)))
            continue
        entry = entries.get(source.name, {})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            findings.append(
                Finding(
                    source.name,
                    "source_hash",
                    "ERROR",
                    len(frame),
                    "Input differs from its manifest or has no manifest entry.",
                )
            )
        frame_findings = inspect_frame(
            frame,
            series=source.name,
            as_of=config.project["end_date"],
            expected_unit=source.unit,
            declared_unit=entry.get("unit"),
            max_age_days=int(config.project["max_staleness_calendar_days"]),
        )
        if "series_id" not in frame.columns or frame["series_id"].isna().any():
            frame_findings.append(
                Finding(
                    source.name,
                    "series_identifier",
                    "ERROR",
                    len(frame),
                    "A nonmissing series identifier is required.",
                )
            )
        findings.extend(frame_findings)
        if not any(x.severity == "ERROR" for x in frame_findings):
            coverage = summarize_weekly_coverage(
                source.name,
                frame,
                config.project["start_date"],
                config.project["end_date"],
                int(config.project["max_staleness_calendar_days"]),
            )
            if coverage.weekly_coverage_percent < 95:
                findings.append(
                    Finding(
                        source.name,
                        "weekly_coverage",
                        "ERROR",
                        0,
                        f"Coverage {coverage.weekly_coverage_percent:.2f}% < 95%.",
                    )
                )
    result = write_findings(findings, report_path)
    if result["status"] == "BLOCKED":
        raise DataQualityError(f"Input controls blocked calculation; inspect {report_path}")
    return result
