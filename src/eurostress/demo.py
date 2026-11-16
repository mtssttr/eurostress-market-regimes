"""Offline, deterministic control investigations using fictional observations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eurostress.controls import inspect_frame, write_findings


def run_demo(directory: str | Path) -> dict:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261116)
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    clean = pd.DataFrame(
        {"date": dates, "value": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))}
    )
    missing = clean.copy()
    missing.loc[10, "value"] = np.nan
    scenarios = {
        "duplicate": (pd.concat([clean, clean.iloc[[8]]], ignore_index=True), "index_points"),
        "missing": (missing, "index_points"),
        "stale": (clean.iloc[:-10].copy(), "index_points"),
        "unit": (clean.copy(), "percent_per_annum"),
    }
    clean.to_csv(root / "synthetic_source.csv", index=False)
    cases = []
    actions = {
        "duplicate": "Verified against source; removed duplicate observation.",
        "missing": "Restored missing value from the synthetic source of truth.",
        "stale": "Reloaded missing dates from the synthetic source of truth.",
        "unit": "Verified source identity and corrected unit metadata; values unchanged.",
    }
    for name, (broken, unit) in scenarios.items():
        broken.to_csv(root / f"{name}_input.csv", index=False)
        before = write_findings(
            inspect_frame(
                broken,
                series="synthetic_equity",
                as_of="2024-03-29",
                expected_unit="index_points",
                declared_unit=unit,
            ),
            root / f"{name}_before.json",
        )
        # Restoring a known synthetic source is a demonstration, not an automatic
        # repair policy for real financial data.
        after = write_findings(
            inspect_frame(
                clean,
                series="synthetic_equity",
                as_of="2024-03-29",
                expected_unit="index_points",
                declared_unit="index_points",
            ),
            root / f"{name}_after.json",
        )
        if before["status"] != "BLOCKED" or after["status"] != "PASS":
            raise RuntimeError(f"Unexpected investigation outcome: {name}")
        cases.append(
            {
                "case": name,
                "before": before["status"],
                "after": after["status"],
                "action": actions[name],
                "calculation_before": "withheld",
                "recomputed_last_return": float(
                    np.log(clean.value.iloc[-1] / clean.value.iloc[-2])
                ),
            }
        )
    pd.DataFrame(cases).to_csv(root / "investigation_log.csv", index=False)
    summary = {
        "classification": "synthetic educational demonstration",
        "cases": cases,
        "source_sha256": hashlib.sha256((root / "synthetic_source.csv").read_bytes()).hexdigest(),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "README.md").write_text(
        "# Offline control demonstration\n\nAll numerical inputs are fictional. "
        "Four deliberately corrupted inputs block the calculation. Each before/after "
        "JSON report and the investigation log document the failure and source-verified "
        "correction. Real-world corrections require investigation; this demo restores "
        "a known synthetic source. Risk outputs are withheld while inputs are blocked.\n"
    )
    return summary
