"""Build a two-page PDF directly from measured project outputs, without network access."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
NAVY = "#18334B"
TEAL = "#147D83"
MUTED = "#536473"


def body(ax, text, x, y, width=95, size=10, color=MUTED):
    lines = "\n".join(textwrap.wrap(text, width=width))
    ax.text(
        x, y, lines, fontsize=size, color=color, va="top", linespacing=1.5, transform=ax.transAxes
    )


def page(number, title, subtitle):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0.085, 0.06, 0.83, 0.88])
    ax.set_axis_off()
    ax.text(
        0,
        1,
        "EUROSTRESS  /  MATIAS DAVID SUTTER",
        fontsize=10,
        color=TEAL,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(0, 0.945, title, fontsize=23, color=NAVY, weight="bold", transform=ax.transAxes)
    body(ax, subtitle, 0, 0.90, size=10)
    fig.text(
        0.085,
        0.035,
        "Independent educational project | AI-assisted | November 2026",
        fontsize=8,
        color=MUTED,
    )
    fig.text(0.90, 0.035, f"{number} / 2", fontsize=8, color=MUTED, ha="right")
    return fig, ax


def main():
    demo = json.loads((ROOT / "reports/demo/summary.json").read_text())
    numerical = json.loads((ROOT / "reports/numerical_validation/checks.json").read_text())
    quality = json.loads((ROOT / "reports/controls/data_quality.json").read_text())
    results = pd.read_csv(ROOT / "reports/tail_extension/comparison.csv")
    primary = results[results.tail_probability.eq(0.01)].copy()
    output = ROOT / "output/pdf"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "EuroStress_Case_Study.pdf"
    with PdfPages(
        target,
        metadata={
            "Title": "EuroStress - Data Controls and Risk Validation",
            "Author": "Matias David Sutter",
        },
    ) as pdf:
        fig, ax = page(
            1,
            "Trust the inputs. Test the model.",
            "A reproducible financial-data workflow with blocking controls, "
            "documented investigations and independent numerical checks.",
        )
        ax.text(0, 0.79, "THE QUESTION", fontsize=10, color=TEAL, weight="bold")
        body(
            ax,
            "Can conventional models describe European equity tail risk reliably, "
            "and can input errors be detected before calculations are released?",
            0,
            0.76,
        )
        ax.text(0, 0.67, "CONTROLLED INVESTIGATION DEMO", fontsize=10, color=TEAL, weight="bold")
        rows = [[x["case"].capitalize(), x["before"], x["after"]] for x in demo["cases"]]
        table = ax.table(
            cellText=rows,
            colLabels=["Synthetic input problem", "Before correction", "After source check"],
            cellLoc="left",
            colLoc="left",
            bbox=[0, 0.43, 1, 0.20],
            colWidths=[0.40, 0.30, 0.30],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (r, _), cell in table.get_celld().items():
            cell.set_edgecolor("#DDE5E8")
            cell.set_facecolor(NAVY if r == 0 else "#F4F8F9")
            cell.set_text_props(color="white" if r == 0 else NAVY)
        body(
            ax,
            "All inputs in the demo are fictional. Calculation is withheld on failure; "
            "the investigation log records the source-verified correction and regenerated return.",
            0,
            0.40,
            size=9,
        )
        ax.text(0, 0.30, "NUMERICAL EVIDENCE", fontsize=10, color=TEAL, weight="bold")
        body(
            ax,
            f"GARCH recursion versus arch: maximum variance difference "
            f"{numerical['garch_max_absolute_variance_error']:.2e}. "
            f"Analytical tail formulas versus quadrature: maximum ES difference "
            f"{numerical['es_max_absolute_quadrature_error']:.2e}. "
            "Independent coverage and independence likelihood calculations agree. "
            f"Across 1,000 known-null simulations, the nominal 5% coverage test rejected "
            f"{numerical['simulation_coverage_rejection_rate']:.1%} of samples.",
            0,
            0.27,
            size=9,
        )
        body(
            ax,
            f"Scope: current source controls pass with {quality['warnings']} explicit "
            "warnings for absent unit declarations in legacy manifests. Metadata matching "
            "does not verify economic correctness. The original research evaluation is preserved.",
            0,
            0.12,
            size=8.7,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = page(
            2,
            "What changed tail calibration?",
            "Exploratory GARCH comparison | EURO STOXX 50 | 1,510 daily forecasts, 2020-2025",
        )
        body(
            ax,
            "Matched design: normal versus unit-variance Student t errors; fixed versus "
            "monthly refitting. Every fit uses the preceding 1,260 returns. All 146 fits "
            "converged. Each forecast excludes the return it predicts.",
            0,
            0.82,
            size=9.5,
        )
        chart = fig.add_axes([0.16, 0.49, 0.73, 0.24])
        labels = ["Gaussian\nfixed", "Gaussian\nmonthly", "Student t\nfixed", "Student t\nmonthly"]
        chart.bar(
            np.arange(4),
            100 * primary.exception_rate,
            color=["#9BAEBB", "#6F8FA3", "#4BABA5", TEAL],
            width=0.6,
        )
        chart.axhline(
            1, color="#B45044", linestyle="--", linewidth=1.2, label="Intended 1% exception rate"
        )
        for i, row in enumerate(primary.itertuples()):
            chart.text(
                i,
                100 * row.exception_rate + 0.08,
                f"{row.exceptions} breaches",
                ha="center",
                fontsize=8.5,
                color=NAVY,
            )
        chart.set_xticks(np.arange(4), labels, fontsize=9)
        chart.set_ylabel("Observed exception rate (%)", fontsize=9)
        chart.set_ylim(0, 3.2)
        chart.spines[["top", "right"]].set_visible(False)
        chart.legend(loc="upper right", fontsize=8, frameon=False)
        ax.text(0, 0.42, "RESULT AND INTERPRETATION", fontsize=10, color=TEAL, weight="bold")
        baseline = primary[primary.model.eq("normal_fixed")].iloc[0]
        candidate = primary[primary.model.eq("t_monthly")].iloc[0]
        reduction = 1 - candidate.mean_quantile_loss / baseline.mean_quantile_loss
        body(
            ax,
            f"Monthly Student t recorded {int(candidate.exceptions)} breaches versus "
            f"{int(baseline.exceptions)} for matched fixed Gaussian, and {reduction:.1%} lower "
            "mean quantile loss. However, 25 breaches still exceeded 15.1 expected; "
            f"correct unconditional coverage was rejected (p={candidate.kupiec_p_value:.3f}). "
            "The extension improved these descriptive measures without resolving undercoverage.",
            0,
            0.39,
            size=9.5,
        )
        body(
            ax,
            "Limits: outcomes were previously examined, so this is exploratory. "
            "No multiplicity-adjusted superiority, trading-profitability or formal "
            "ES-validation claim is made. Historical revision vintages have not been fully "
            "reconstructed. This new comparator differs from the original GARCH baseline.",
            0,
            0.24,
            size=8.7,
        )
        body(
            ax,
            "Reproduce: make demo numerical-validation data-gate tail-extension case-study. "
            "The historical comparison requires local market data; the synthetic demo and "
            "numerical checks work offline after dependency installation.",
            0,
            0.13,
            size=8.7,
        )
        body(
            ax,
            "Methods: arch documentation; Kupiec (1995); Christoffersen (1998). "
            "Full source references and configuration hashes are in the repository reports.",
            0,
            0.045,
            size=8,
        )
        pdf.savefig(fig)
        plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
