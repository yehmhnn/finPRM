#!/usr/bin/env python3
"""Generate release figures from the committed result tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


COLORS = {"Base": "#4C78A8", "LoRA": "#F58518"}


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_figure(figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Creator": "scripts/plot_results.py", "Date": None},
    )
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )


def grouped_bars(
    rows: Sequence[Mapping[str, str]],
    categories: Sequence[str],
    category_key: str,
    metric_key: str,
    labels: Sequence[str],
    title: str,
    ylabel: str,
    output_dir: Path,
    stem: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    values = {
        (row["model"], row[category_key]): float(row[metric_key]) for row in rows
    }
    x = np.arange(len(categories))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for model, offset in (("Base", -width / 2), ("LoRA", width / 2)):
        heights = [values[(model, category)] for category in categories]
        bars = axis.bar(
            x + offset,
            heights,
            width,
            label=model,
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.7,
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in heights], padding=3)

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.22)
    save_figure(figure, output_dir, stem)
    plt.close(figure)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=project_root / "results")
    parser.add_argument(
        "--output", type=Path, default=project_root / "assets/figures"
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "svg.hashsalt": "finprm-release-v1",
        }
    )

    main_rows = load_csv(args.results / "main_metrics.csv")
    grouped_bars(
        main_rows,
        ("none", "question", "joint"),
        "retrieval_key",
        "test_macro_f1",
        ("No retrieval", "Question-only", "Joint"),
        "Macro-F1 across adaptation conditions",
        "Test Macro-F1",
        args.output,
        "main_macro_f1",
    )

    step_rows = load_csv(args.results / "step_position_metrics.csv")
    grouped_bars(
        step_rows,
        ("Step 1", "Step 2", "Step 3+"),
        "position",
        "macro_f1",
        ("Step 1", "Step 2", "Step 3+"),
        "Macro-F1 by candidate-step position",
        "Test Macro-F1",
        args.output,
        "step_position_macro_f1",
    )

    corruption_rows = load_csv(args.results / "corruption_type_metrics.csv")
    corruption_types = (
        "dangling_reference",
        "entity_context_swap",
        "operand_reversal",
        "operator_substitution",
        "unit_scale_mismatch",
    )
    grouped_bars(
        corruption_rows,
        corruption_types,
        "corruption_type",
        "accuracy",
        (
            "Dangling\nreference",
            "Entity/context\nswap",
            "Operand\nreversal",
            "Operator\nsubstitution",
            "Unit/scale\nmismatch",
        ),
        "Negative detection by corruption type",
        "Accuracy on negative examples",
        args.output,
        "corruption_type_accuracy",
    )


if __name__ == "__main__":
    main()
