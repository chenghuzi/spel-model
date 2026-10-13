import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mannwhitneyu

from analysis.numerosity import (
    build_match_sequences,
    build_model_for_checkpoint,
    get_analysis_mask,
    get_time_masks,
    load_config,
    resolve_device,
    run_model,
)
from analysis.population import jsonable_phase, phase_population
from models.storage import PUBLISHED_CHECKPOINT, list_checkpoints, parse_epoch
from paths import ARTIFACTS, ROOT, output_directory
from visualizations.style import NUMBER_COLORS


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot Nieder/Miller-style population numerosity tuning for a trained "
            "numerosity RNN checkpoint."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/numerosity.json")
    parser.add_argument("--checkpoint", type=Path, default=PUBLISHED_CHECKPOINT)
    parser.add_argument("--checkpoint-glob", default=None)
    parser.add_argument("--epochs", nargs="*", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--min-checkpoint-age", type=float, default=20.0)
    parser.add_argument("--stimulus-mode", default="standard")
    parser.add_argument("--trials-per-numerosity", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--cell-threshold",
        choices=["fdr", "uncorrected"],
        default="fdr",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS / "numerosity" / "tuning",
    )
    return parser.parse_args()


def p_to_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def add_adjacent_distance_stats(ax, distance_values, by_distance):
    pairs = [(-4, -3), (-3, -2), (-2, -1), (1, 2), (2, 3), (3, 4)]
    y_base = 104.0
    y_step = 6.0
    used = 0
    for left, right in pairs:
        if left not in by_distance or right not in by_distance:
            continue
        left_values = np.asarray(by_distance[left], dtype=float)
        right_values = np.asarray(by_distance[right], dtype=float)
        if len(left_values) < 3 or len(right_values) < 3:
            continue
        alternative = "less" if left < 0 else "greater"
        _, p_value = mannwhitneyu(left_values, right_values, alternative=alternative)
        stars = p_to_stars(float(p_value))
        if not stars:
            continue
        y = y_base + y_step * (used % 2)
        ax.plot([left, left, right, right], [y - 1.8, y, y, y - 1.8], color="black", linewidth=1.6)
        ax.text(
            (left + right) / 2.0,
            y + 1.4,
            stars,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        used += 1


def format_axes(ax):
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.8)
    ax.tick_params(axis="both", width=1.6, length=4, labelsize=10)


def plot_distance_panel(ax, phase, title=None):
    ax.errorbar(
        phase["distance_values"],
        phase["distance_mean"],
        yerr=phase["distance_sem"],
        color="#1a1a1a",
        marker="o",
        markersize=5.2,
        linewidth=2.0,
        capsize=2.0,
    )
    add_adjacent_distance_stats(ax, phase["distance_values"], phase["by_distance"])
    ax.set_xlim(-4.35, 4.35)
    ax.set_ylim(0, 118)
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Numerical distance", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized\nresponse (%)", fontsize=12, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    format_axes(ax)


def plot_group_tuning_panel(ax, phase, numerosities):
    for idx, numerosity in enumerate(numerosities):
        y = phase["tuning_mean"][idx]
        yerr = phase["tuning_sem"][idx]
        if np.isnan(y).all():
            continue
        ax.errorbar(
            numerosities,
            y,
            yerr=yerr,
            color=NUMBER_COLORS[idx % len(NUMBER_COLORS)],
            marker="o",
            markersize=4.8,
            linewidth=2.2,
            capsize=1.8,
        )
    ax.set_xlim(numerosities.min() - 0.25, numerosities.max() + 0.25)
    ax.set_ylim(0, 105)
    ax.set_xticks(numerosities)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Number of items", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized\nresponse (%)", fontsize=12, fontweight="bold")
    format_axes(ax)


def panel_letter(ax, letter):
    ax.text(
        -0.20,
        1.08,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=20,
        fontweight="bold",
    )


def plot_population_figure(sample_phase, delay_phase, numerosities, output_png, output_pdf):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
    plot_distance_panel(axes[0, 0], sample_phase, title="Sample period")
    plot_distance_panel(axes[0, 1], delay_phase, title="Delay period")
    plot_group_tuning_panel(axes[1, 0], sample_phase, numerosities)
    plot_group_tuning_panel(axes[1, 1], delay_phase, numerosities)

    for letter, ax in zip("ABCD", axes.flat):
        panel_letter(ax, letter)

    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    config, sim_config, numerosity_config, metadata = load_config(args.config)
    checkpoint = max(list_checkpoints(args), key=parse_epoch)
    epoch = parse_epoch(checkpoint)
    masks = get_time_masks(sim_config, numerosity_config, metadata)
    args.output_dir = output_directory(args.output_dir)

    min_n = int(numerosity_config["min_numerosity"])
    max_n = int(numerosity_config["max_numerosity"])
    numerosities = np.arange(min_n, max_n + 1)

    print(f"loading checkpoint epoch {epoch}: {checkpoint}")
    model = build_model_for_checkpoint(config, sim_config, metadata, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    match_sequences, labels = build_match_sequences(
        numerosity_config,
        metadata,
        masks,
        args.stimulus_mode,
        args.trials_per_numerosity,
        seed=args.seed,
    )
    hiddens, _ = run_model(
        model,
        match_sequences,
        args.batch_size,
        device,
        seed=args.seed + epoch * 17,
    )
    sample_features = hiddens[:, get_analysis_mask(masks, "sample"), :].mean(axis=1)
    delay_features = hiddens[:, get_analysis_mask(masks, "delay"), :].mean(axis=1)
    sample_phase = phase_population(
        sample_features,
        labels,
        numerosities,
        args.alpha,
        args.cell_threshold,
    )
    delay_phase = phase_population(
        delay_features,
        labels,
        numerosities,
        args.alpha,
        args.cell_threshold,
    )

    stem = "tuning"
    output_png = args.output_dir / f"{stem}.png"
    output_pdf = args.output_dir / f"{stem}.pdf"
    output_json = args.output_dir / f"{stem}.json"
    plot_population_figure(
        sample_phase,
        delay_phase,
        numerosities,
        output_png,
        output_pdf,
    )

    payload = {
        "schema_version": 1,
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "epoch": int(epoch),
        "stimulus_mode": args.stimulus_mode,
        "trials_per_numerosity": int(args.trials_per_numerosity),
        "alpha": float(args.alpha),
        "cell_threshold": args.cell_threshold,
        "sample": jsonable_phase(sample_phase),
        "delay": jsonable_phase(delay_phase),
        "outputs": {
            "png": str(output_png),
            "pdf": str(output_pdf),
            "json": str(output_json),
        },
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
