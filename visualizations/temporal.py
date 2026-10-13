import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import linregress, spearmanr

from analysis import numerosity as cells
from analysis.cache import (
    ANALYSIS_MODES,
    analysis_fingerprint,
    compute_model_analysis,
    load_analysis_cache,
    save_analysis_cache,
)
from analysis.numerosity import resolve_device
from paths import ARTIFACTS, ROOT, output_directory
from visualizations.style import MODE_COLORS, MODE_LABELS, NUMBER_COLORS

CATEGORIES = [
    ("sample_positive", "sample", 1, "Sample-positive"),
    ("sample_negative", "sample", -1, "Sample-negative"),
    ("delay_positive", "delay", 1, "Delay-positive"),
    ("delay_negative", "delay", -1, "Delay-negative"),
]
PERIOD_COLORS = {"sample": "#111111", "delay": "#8A8A8A"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find sample- and delay-dominant monotonic numerosity units and "
            "render manuscript-style examples."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/numerosity.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--trials-per-numerosity", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--selected-units",
        type=int,
        nargs=4,
        default=None,
        metavar=("SAMPLE_POS", "SAMPLE_NEG", "DELAY_POS", "DELAY_NEG"),
        help="Override automatic selections in category order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS / "numerosity/temporal",
    )
    parser.add_argument("--force-analysis", action="store_true")
    return parser.parse_args()


def phase_metrics(profile):
    numerosities = np.arange(1, len(profile) + 1, dtype=float)
    regression = linregress(numerosities, profile)
    rho = spearmanr(numerosities, profile).statistic
    return {
        "slope": float(regression.slope),
        "r2": float(regression.rvalue**2),
        "p": float(regression.pvalue),
        "rho": float(rho),
        "range": float(np.ptp(profile)),
        "mean": float(np.mean(profile)),
    }


def score_units(summaries):
    standard = summaries["standard"]
    profiles = {
        "sample": standard["sample_mean_activity"].T,
        "delay": standard["delay_mean_activity"].T,
    }
    metrics = {
        phase: [phase_metrics(profile) for profile in phase_profiles]
        for phase, phase_profiles in profiles.items()
    }
    phase_ranges = {
        phase: np.asarray([row["range"] for row in phase_metrics_rows])
        for phase, phase_metrics_rows in metrics.items()
    }
    range_scales = {
        phase: max(float(np.quantile(values, 0.75)), 1e-8) for phase, values in phase_ranges.items()
    }

    rankings = {}
    for key, target_phase, direction, _ in CATEGORIES:
        other_phase = "delay" if target_phase == "sample" else "sample"
        rows = []
        for unit in range(profiles[target_phase].shape[0]):
            target = metrics[target_phase][unit]
            other = metrics[other_phase][unit]
            target_linearity = target["r2"] * abs(target["rho"])
            other_linearity = other["r2"] * abs(other["rho"])
            target_visibility = np.log1p(target["range"] / range_scales[target_phase])
            other_visibility = np.log1p(other["range"] / range_scales[other_phase])
            target_strength = target_linearity * target_visibility
            other_strength = other_linearity * other_visibility
            sign_match = direction * target["slope"] > 0
            score = (
                target_strength
                + 1.25 * (target_strength - other_strength)
                + 0.5 * target["r2"]
                + 0.35 * abs(target["rho"])
            )
            if not sign_match:
                score -= 10.0
            rows.append(
                {
                    "category": key,
                    "unit": unit,
                    "score": float(score),
                    "target_phase": target_phase,
                    "direction": direction,
                    "target_strength": float(target_strength),
                    "other_strength": float(other_strength),
                    "dominance": float(target_strength / (target_strength + other_strength + 1e-8)),
                    "sample_slope": metrics["sample"][unit]["slope"],
                    "sample_r2": metrics["sample"][unit]["r2"],
                    "sample_rho": metrics["sample"][unit]["rho"],
                    "sample_range": metrics["sample"][unit]["range"],
                    "delay_slope": metrics["delay"][unit]["slope"],
                    "delay_r2": metrics["delay"][unit]["r2"],
                    "delay_rho": metrics["delay"][unit]["rho"],
                    "delay_range": metrics["delay"][unit]["range"],
                    "sample_profile": profiles["sample"][unit].tolist(),
                    "delay_profile": profiles["delay"][unit].tolist(),
                }
            )
        rankings[key] = sorted(rows, key=lambda row: row["score"], reverse=True)
    return rankings


def select_units(rankings):
    selected = {}
    used = set()
    for key, _, _, _ in CATEGORIES:
        for row in rankings[key]:
            if row["unit"] in used:
                continue
            selected[key] = int(row["unit"])
            used.add(row["unit"])
            break
    return selected


def validate_selected_units(selected, summaries):
    validation = {}
    for key, phase, _, _ in CATEGORIES:
        unit = selected[key]
        standard_profile = summaries["standard"][f"{phase}_mean_activity"][:, unit]
        mode_metrics = {}
        for mode in ANALYSIS_MODES:
            profile = summaries[mode][f"{phase}_mean_activity"][:, unit]
            metrics = phase_metrics(profile)
            # Constant profiles have undefined correlation, represented by NumPy's NaN.
            profile_correlation = float(np.corrcoef(standard_profile, profile)[0, 1])
            mode_metrics[mode] = {
                **metrics,
                "profile_correlation_with_standard": profile_correlation,
                "profile": profile.tolist(),
            }
        validation[key] = {
            "unit": unit,
            "phase": phase,
            "modes": mode_metrics,
        }
    return validation


def style_axis(ax):
    ax.tick_params(labelsize=8, width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def add_phase_bands(ax, masks):
    ax.axvspan(
        masks["sample_start"],
        masks["sample_end"],
        color="#D9DDE3",
        alpha=0.9,
        zorder=-3,
    )
    ax.axvspan(
        masks["sample_end"],
        masks["test_start"],
        color="#EEF0F3",
        alpha=0.9,
        zorder=-3,
    )


def add_temporal_unit_panel(
    ax,
    unit,
    title,
    summaries,
    masks,
    numerosity_config,
    show_legend=False,
    compact=False,
    standalone=False,
    inset_loc="upper right",
    inset_bbox_to_anchor=None,
    legend_loc="upper left",
    inset_periods=("sample", "delay"),
    inset_size=None,
    inset_show_ylabel=True,
):
    standard = summaries["standard"]
    times = masks["times"]
    plot_mask = masks["plot_mask"]
    numerosities = np.arange(
        int(numerosity_config["min_numerosity"]),
        int(numerosity_config["max_numerosity"]) + 1,
    )
    if compact:
        line_width = 0.9
        title_size = 7
        axis_label_size = 6
        tick_size = 5
        inset_width = "40%"
        inset_height = "38%"
        inset_tick_size = 4
        inset_label_size = 6
        inset_legend_size = 5.5
    elif standalone:
        line_width = 2.0
        title_size = 12
        axis_label_size = 11
        tick_size = 9
        inset_width = "44%"
        inset_height = "40%"
        inset_tick_size = 7
        inset_label_size = 7.5
        inset_legend_size = 7
    else:
        line_width = 1.8
        title_size = 11
        axis_label_size = 10
        tick_size = 8
        inset_width = "40%"
        inset_height = "38%"
        inset_tick_size = 6
        inset_label_size = 6
        inset_legend_size = 5.5

    if inset_size is not None:
        inset_width, inset_height = inset_size

    add_phase_bands(ax, masks)
    for num_index, numerosity in enumerate(numerosities):
        ax.plot(
            times[plot_mask],
            standard["mean_time"][num_index, plot_mask, unit],
            color=NUMBER_COLORS[num_index],
            linewidth=line_width,
            label=str(int(numerosity)),
        )
    ax.set_xlim(0, masks["plot_end"])
    panel_title = f"{title}  |  neuron #{unit}" if title else f"neuron #{unit}"
    ax.set_title(
        panel_title,
        loc="left",
        fontsize=title_size,
    )
    ax.set_xlabel("Time (ms)", fontsize=axis_label_size)
    ax.set_ylabel("Activity", fontsize=axis_label_size)
    ax.tick_params(labelsize=tick_size, length=2 if compact else 3)
    if show_legend:
        ax.legend(
            title="N",
            frameon=False,
            fontsize=8 if standalone else 7,
            title_fontsize=9 if standalone else 8,
            handlelength=1.3,
            loc=legend_loc,
        )

    inset_kwargs = {}
    if inset_bbox_to_anchor is not None:
        inset_kwargs = {
            "bbox_to_anchor": inset_bbox_to_anchor,
            "bbox_transform": ax.transAxes,
        }
    inset = inset_axes(
        ax,
        width=inset_width,
        height=inset_height,
        loc=inset_loc,
        borderpad=0.7 if compact else 0.9,
        **inset_kwargs,
    )
    for phase, marker, linestyle in [
        ("sample", "o", "-"),
        ("delay", "s", "--"),
    ]:
        if phase not in inset_periods:
            continue
        values = standard[f"{phase}_mean_activity"][:, unit]
        errors = standard[f"{phase}_sem_activity"][:, unit]
        inset.errorbar(
            numerosities,
            values,
            yerr=errors,
            color=PERIOD_COLORS[phase],
            linewidth=0.65 if compact else 1.15,
            linestyle=linestyle,
            marker=marker,
            markersize=1.8 if compact else 3.0,
            capsize=0.8 if compact else 1.4,
            label=phase,
        )
    inset.set_xticks(numerosities)
    inset.tick_params(labelsize=inset_tick_size, length=1.5)
    if not compact:
        inset.set_xlabel("Number of items", fontsize=inset_label_size)
        if inset_show_ylabel:
            inset.set_ylabel("Mean activity", fontsize=inset_label_size)
        if len(inset_periods) > 1:
            inset.legend(
                frameon=False,
                fontsize=inset_legend_size,
                handlelength=1.2,
                loc="best",
            )
    sns.despine(ax=ax)
    sns.despine(ax=inset)


def render_selected_examples(
    path,
    selected,
    summaries,
    masks,
    numerosity_config,
):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5))
    for panel_index, (ax, (_, _, _, title)) in enumerate(zip(axes.flat, CATEGORIES)):
        key = CATEGORIES[panel_index][0]
        add_temporal_unit_panel(
            ax,
            selected[key],
            title,
            summaries,
            masks,
            numerosity_config,
            show_legend=panel_index == 0,
        )
        ax.text(
            -0.14,
            1.06,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
        )
        style_axis(ax)
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    sns.despine()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_individual_examples(
    output_dir,
    selected,
    summaries,
    masks,
    numerosity_config,
):
    outputs = {}
    for key, phase, _, _ in CATEGORIES:
        unit = selected[key]
        stem = f"unit{unit:03d}"
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        add_temporal_unit_panel(
            ax,
            unit,
            "",
            summaries,
            masks,
            numerosity_config,
            show_legend=key == "sample_positive",
            standalone=True,
            inset_loc="upper left" if phase == "delay" else "upper right",
            inset_bbox_to_anchor=((0.10, 0.0, 0.90, 1.0) if phase == "delay" else None),
            legend_loc="lower left" if phase == "delay" else "upper left",
        )
        fig.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.88)
        fig.savefig(png_path, dpi=100, facecolor="white")
        fig.savefig(pdf_path, facecolor="white")
        plt.close(fig)
        outputs[key] = {
            "unit": unit,
            "png": str(png_path),
            "pdf": str(pdf_path),
        }
    return outputs


def render_individual_control_examples(
    output_dir,
    selected,
    summaries,
    numerosity_config,
):
    numerosities = np.arange(
        int(numerosity_config["min_numerosity"]),
        int(numerosity_config["max_numerosity"]) + 1,
    )
    outputs = {}
    for key, phase, direction, _ in CATEGORIES:
        unit = selected[key]
        stem = f"ctrl{unit:03d}"
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        for mode in ANALYSIS_MODES:
            ax.errorbar(
                numerosities,
                summaries[mode][f"{phase}_mean_activity"][:, unit],
                yerr=summaries[mode][f"{phase}_sem_activity"][:, unit],
                color=MODE_COLORS[mode],
                linewidth=1.6,
                marker="o",
                markersize=4.0,
                capsize=1.5,
                label=MODE_LABELS[mode],
            )
        ax.set_title(f"neuron #{unit}", loc="left", fontsize=12)
        ax.set_xticks(numerosities)
        ax.set_xlabel("Number of items", fontsize=11)
        ax.set_ylabel("Activity", fontsize=11)
        ax.tick_params(labelsize=9, width=0.8, length=3)
        ax.margins(x=0.04, y=0.08)
        ax.legend(
            frameon=False,
            fontsize=6.5,
            ncol=2,
            loc="upper left" if direction > 0 else "lower left",
            handlelength=1.2,
            handletextpad=0.35,
            columnspacing=0.7,
            labelspacing=0.3,
            borderaxespad=0.4,
        )
        sns.despine(ax=ax)
        fig.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.88)
        fig.savefig(png_path, dpi=100, facecolor="white")
        fig.savefig(pdf_path, facecolor="white")
        plt.close(fig)
        outputs[key] = {
            "unit": unit,
            "phase": phase,
            "png": str(png_path),
            "pdf": str(pdf_path),
        }
    return outputs


def render_control_validation(
    path,
    selected,
    summaries,
    numerosity_config,
):
    numerosities = np.arange(
        int(numerosity_config["min_numerosity"]),
        int(numerosity_config["max_numerosity"]) + 1,
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5))
    for panel_index, (ax, (key, phase, _, title)) in enumerate(zip(axes.flat, CATEGORIES)):
        unit = selected[key]
        for mode in ANALYSIS_MODES:
            ax.errorbar(
                numerosities,
                summaries[mode][f"{phase}_mean_activity"][:, unit],
                yerr=summaries[mode][f"{phase}_sem_activity"][:, unit],
                color=MODE_COLORS[mode],
                linewidth=1.25,
                marker="o",
                markersize=3.0,
                capsize=1.2,
                label=MODE_LABELS[mode],
            )
        ax.set_title(f"{title}  |  neuron #{unit}", loc="left", fontsize=11)
        ax.set_xticks(numerosities)
        ax.set_xlabel("Number of items", fontsize=10)
        ax.set_ylabel("Activity", fontsize=10)
        ax.text(
            -0.14,
            1.06,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
        )
        style_axis(ax)
        sns.despine(ax=ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.985, 0.5),
        fontsize=8,
        handlelength=1.4,
    )
    fig.tight_layout(rect=(0, 0, 0.84, 1), w_pad=2.0, h_pad=2.0)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir = output_directory(args.output_dir)
    device = resolve_device(args.device)
    config, sim_config, numerosity_config, metadata = cells.load_config(args.config)
    masks = cells.get_time_masks(sim_config, numerosity_config, metadata)
    # Reuse the same activity collection and cache format as the population figures.
    fingerprint = analysis_fingerprint(args, args.checkpoint)
    cache_path = args.output_dir / "cache.npz"
    analysis = None if args.force_analysis else load_analysis_cache(cache_path, fingerprint)
    cache_hit = analysis is not None
    if analysis is None:
        analysis = compute_model_analysis(
            args, config, sim_config, numerosity_config, metadata, masks, args.checkpoint, device
        )
        save_analysis_cache(cache_path, fingerprint, analysis)
    summaries = analysis["summaries"]

    rankings = score_units(summaries)
    if args.selected_units is None:
        selected = select_units(rankings)
    else:
        selected = {key: int(unit) for (key, _, _, _), unit in zip(CATEGORIES, args.selected_units)}
    control_validation = validate_selected_units(selected, summaries)

    examples_path = args.output_dir / "examples.png"
    controls_path = args.output_dir / "controls.png"
    render_selected_examples(
        examples_path,
        selected,
        summaries,
        masks,
        numerosity_config,
    )
    individual_examples = render_individual_examples(
        args.output_dir,
        selected,
        summaries,
        masks,
        numerosity_config,
    )
    individual_control_examples = render_individual_control_examples(
        args.output_dir,
        selected,
        summaries,
        numerosity_config,
    )
    render_control_validation(
        controls_path,
        selected,
        summaries,
        numerosity_config,
    )
    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "trials_per_numerosity": args.trials_per_numerosity,
        "cache_hit": cache_hit,
        "fingerprint": fingerprint,
        "selected_units": selected,
        "selected_metrics": {
            key: next(row for row in rankings[key] if row["unit"] == selected[key])
            for key, _, _, _ in CATEGORIES
        },
        "control_validation": control_validation,
        "outputs": {
            "examples_png": str(examples_path),
            "examples_pdf": str(examples_path.with_suffix(".pdf")),
            "individual_examples": individual_examples,
            "individual_control_examples": individual_control_examples,
            "controls_png": str(controls_path),
            "controls_pdf": str(controls_path.with_suffix(".pdf")),
        },
    }
    metadata_path = args.output_dir / "analysis.json"
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
