#!/usr/bin/env python3
"""Plot the recorded closed-loop accuracy of the seven visual controls."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from paths import ARTIFACTS, output_directory
from visualizations.style import MODE_COLORS

CONTROL_MODES = [
    "equal_area",
    "equal_circumference",
    "shape",
    "linear",
    "high_density",
    "low_density",
    "variable_features",
]

CONTROL_LABELS = {
    "equal_area": "Equal\narea",
    "equal_circumference": "Equal\ncircum.",
    "shape": "Shape",
    "linear": "Linear",
    "high_density": "High\ndensity",
    "low_density": "Low\ndensity",
    "variable_features": "Variable\nfeatures",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-summary",
        type=Path,
        default=ARTIFACTS / "numerosity/published/summary.json",
    )
    parser.add_argument(
        "--linear-accuracy",
        type=float,
        default=0.814,
        help="Accuracy from the current arbitrary-orientation linear renderer.",
    )
    parser.add_argument("--linear-trials", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS / "numerosity/controls",
    )
    return parser.parse_args()


def wilson_interval(correct, total, z=1.959963984540054):
    proportion = correct / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half_width = (
        z * np.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)) / denominator
    )
    return center - half_width, center + half_width


def load_results(args):
    summary = json.loads(args.wandb_summary.read_text())
    results = []
    for mode in CONTROL_MODES:
        if mode == "linear":
            accuracy = args.linear_accuracy
            trials = args.linear_trials
            source = "current-renderer closed-loop evaluation"
        else:
            accuracy = float(summary[f"eval/{mode}/correct_rate"])
            trials = int(summary[f"eval/{mode}/closed_loop_trials"])
            source = str(args.wandb_summary)
        correct = int(round(accuracy * trials))
        results.append(
            {
                "mode": mode,
                "accuracy": accuracy,
                "trials": trials,
                "correct": correct,
                "source": source,
            }
        )
    return results


def render(results, output_dir):
    output_dir = output_directory(output_dir)
    modes = [result["mode"] for result in results]
    values = np.asarray([result["accuracy"] for result in results]) * 100.0
    intervals = (
        np.asarray([wilson_interval(result["correct"], result["trials"]) for result in results])
        * 100.0
    )
    errors = np.vstack([values - intervals[:, 0], intervals[:, 1] - values])
    positions = np.arange(len(modes))

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    bars = ax.bar(
        positions,
        values,
        yerr=errors,
        width=0.72,
        color=[MODE_COLORS[mode] for mode in modes],
        edgecolor="none",
        error_kw={
            "ecolor": "#333333",
            "elinewidth": 0.9,
            "capsize": 2.0,
            "capthick": 0.9,
        },
        zorder=2,
    )
    ax.axhline(
        50.0,
        color="#777777",
        linewidth=1.0,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    ax.text(
        len(modes) - 0.45,
        52.0,
        "chance",
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=7,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(value + 3.0, 106.0),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ax.set_title("Control accuracy", loc="left", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(
        positions,
        [CONTROL_LABELS[mode] for mode in modes],
        fontsize=8,
    )
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylim(0, 110)
    ax.set_xlim(-0.6, len(modes) - 0.4)
    ax.tick_params(axis="y", labelsize=9, width=0.8, length=3)
    ax.tick_params(axis="x", width=0.8, length=3, pad=2)
    sns.despine(ax=ax)
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.28, top=0.84)

    png_path = output_dir / "accuracy.png"
    pdf_path = output_dir / "accuracy.pdf"
    fig.savefig(png_path, dpi=100, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    args.output_dir = output_directory(args.output_dir)
    results = load_results(args)
    png_path, pdf_path = render(results, args.output_dir)
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(
        json.dumps(
            {
                "png": str(png_path),
                "pdf": str(pdf_path),
                "results": str(results_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
