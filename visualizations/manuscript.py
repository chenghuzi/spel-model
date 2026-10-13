import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Keep figure generation from silently becoming a second CPU training job.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

from analysis import numerosity as cells
from analysis import population
from analysis.cache import (
    analysis_fingerprint,
    compute_model_analysis,
    load_analysis_cache,
    save_analysis_cache,
)
from models.storage import PUBLISHED_CHECKPOINT, list_checkpoints, parse_epoch
from paths import ARTIFACTS, ROOT, file_sha256, output_directory
from visualizations import population as population_plots
from visualizations.style import MODE_LABELS, NUMBER_COLORS

PAPER_CITATION = (
    "Nieder A, Freedman DJ, Miller EK. Representation of the quantity of visual "
    "items in the primate prefrontal cortex. Science. 2002;297:1708-1711."
)
PAPER_DOI = "10.1126/science.1072493"
DEFAULT_PAPER_PDF = ARTIFACTS / "numerosity/assets/nieder2002.pdf"
DEFAULT_OUTPUT_DIR = ARTIFACTS / "numerosity/population"

# Crop coordinates are PDF points measured from the top-left corner of the page.
# Figure 2 panels are cropped independently to omit the unrelated recording sites.
DEFAULT_SINGLE_PANEL_CROPS = {
    "B": (430.0, 262.0, 560.0, 377.0),
    "C": (315.0, 370.0, 430.0, 484.0),
    "D": (432.0, 370.0, 560.0, 484.0),
    "E": (315.0, 478.0, 430.0, 597.0),
    "F": (432.0, 478.0, 560.0, 597.0),
}
DEFAULT_POPULATION_CROP = (146.0, 64.0, 398.0, 246.0)

PANEL_SPECS = {
    "B": {"controls": ["variable_features"]},
    "C": {"controls": ["high_density", "low_density"]},
    "D": {"controls": ["linear", "shape"]},
    "E": {"controls": ["standard", "equal_area"]},
    "F": {"controls": ["standard", "equal_circumference"]},
}
CONTROL_STYLES = [
    {"color": "#111111", "markerfacecolor": "#111111"},
    {"color": "#7b8488", "markerfacecolor": "white"},
    {"color": "#4d4d4d", "markerfacecolor": "#4d4d4d"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build manuscript-ready side-by-side comparisons of Nieder et al. "
            "Figures 2B-F and 3A-D with a trained numerosity RNN."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/numerosity.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PUBLISHED_CHECKPOINT,
        help="Model checkpoint; defaults to the preserved manuscript model.",
    )
    parser.add_argument("--checkpoint-glob", default=None)
    parser.add_argument("--epochs", nargs="*", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--min-checkpoint-age", type=float, default=20.0)
    parser.add_argument(
        "--sample-units",
        type=int,
        nargs=5,
        default=[8, 68, 66, 53, 18],
        metavar=("B", "C", "D", "E", "F"),
        help="Sample-period model units corresponding to source panels B through F.",
    )
    parser.add_argument(
        "--delay-units",
        type=int,
        nargs=5,
        default=[65, 55, 67, 10, 14],
        metavar=("B", "C", "D", "E", "F"),
        help="Delay-period model units corresponding to source panels B through F.",
    )
    parser.add_argument("--trials-per-numerosity", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--cell-threshold",
        choices=["fdr", "uncorrected"],
        default="fdr",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--paper-pdf", type=Path, default=DEFAULT_PAPER_PDF)
    parser.add_argument("--paper-single-page", type=int, default=3)
    parser.add_argument("--paper-population-page", type=int, default=4)
    parser.add_argument(
        "--single-panel-crop",
        action="append",
        default=[],
        metavar="PANEL:X0,Y0,X1,Y1",
        help="Override a Figure 2 crop in top-left PDF points; may be repeated.",
    )
    parser.add_argument(
        "--population-crop",
        type=float,
        nargs=4,
        default=DEFAULT_POPULATION_CROP,
        metavar=("X0", "Y0", "X1", "Y1"),
    )
    parser.add_argument("--source-dpi", type=int, default=600)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--analysis-cache",
        type=Path,
        default=None,
        help="Optional NPZ cache path. Defaults to OUTPUT_DIR/cache.npz.",
    )
    parser.add_argument("--force-analysis", action="store_true")
    return parser.parse_args()


def parse_crop_overrides(values):
    crops = dict(DEFAULT_SINGLE_PANEL_CROPS)
    for value in values:
        try:
            panel, coordinates = value.split(":", maxsplit=1)
            box = tuple(float(item) for item in coordinates.split(","))
        except ValueError as exc:
            raise ValueError(f"invalid crop {value!r}; expected PANEL:X0,Y0,X1,Y1") from exc
        panel = panel.upper()
        if panel not in crops or len(box) != 4:
            raise ValueError(
                f"invalid crop {value!r}; panel must be B-F and four coordinates are required"
            )
        validate_crop(box, f"panel {panel}")
        crops[panel] = box
    return crops


def validate_crop(box, label):
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0 or min(box) < 0:
        raise ValueError(f"invalid {label} crop: {box}")


def load_or_compute_analysis(
    args,
    config,
    sim_config,
    numerosity_config,
    metadata,
    masks,
    checkpoint,
    device,
    cache_path,
):
    fingerprint = analysis_fingerprint(args, checkpoint)
    if not args.force_analysis:
        analysis = load_analysis_cache(cache_path, fingerprint)
        if analysis is not None:
            print(f"using model-analysis cache: {cache_path}")
            return analysis, fingerprint, True
    analysis = compute_model_analysis(
        args,
        config,
        sim_config,
        numerosity_config,
        metadata,
        masks,
        checkpoint,
        device,
    )
    save_analysis_cache(cache_path, fingerprint, analysis)
    return analysis, fingerprint, False


def render_pdf_page(pdf_path, page_number, dpi):
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise RuntimeError("pdftoppm is required to render the reference PDF")
    with tempfile.TemporaryDirectory(prefix="nieder-paper-") as tmp_dir:
        prefix = Path(tmp_dir) / "page"
        command = [
            executable,
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-r",
            str(dpi),
            "-png",
            "-singlefile",
            str(pdf_path),
            str(prefix),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        with Image.open(prefix.with_suffix(".png")) as image:
            return image.convert("RGB")


def crop_pdf_points(page_image, box, dpi):
    validate_crop(box, "PDF")
    scale = dpi / 72.0
    pixel_box = tuple(int(round(value * scale)) for value in box)
    page_box = (0, 0, page_image.width, page_image.height)
    if (
        pixel_box[0] < page_box[0]
        or pixel_box[1] < page_box[1]
        or pixel_box[2] > page_box[2]
        or pixel_box[3] > page_box[3]
    ):
        raise ValueError(f"crop {box} exceeds rendered PDF page bounds")
    return page_image.crop(pixel_box)


def remove_neighboring_panel_label(image, panel, dpi):
    cleaned = image.copy()
    if panel in {"D", "E", "F"}:
        patch_height = int(round(7.0 * dpi / 72.0))
        x0 = int(round(cleaned.width * 0.26))
        x1 = int(round(cleaned.width * 0.76))
        cleaned.paste("white", (x0, 0, x1, patch_height))
    if panel in {"B", "C", "D"}:
        patch_width = int(round(15.0 * dpi / 72.0))
        patch_height = int(round(8.0 * dpi / 72.0))
        cleaned.paste(
            "white",
            (0, cleaned.height - patch_height, patch_width, cleaned.height),
        )
    return cleaned


def extract_reference_panels(args, single_crops):
    args.output_dir = output_directory(args.output_dir)
    single_page = render_pdf_page(
        args.paper_pdf,
        args.paper_single_page,
        args.source_dpi,
    )
    single_panels = {}
    for panel, box in single_crops.items():
        image = crop_pdf_points(single_page, box, args.source_dpi)
        image = remove_neighboring_panel_label(image, panel, args.source_dpi)
        image.save(args.output_dir / f"ref_{panel}.png")
        single_panels[panel] = image

    population_page = render_pdf_page(
        args.paper_pdf,
        args.paper_population_page,
        args.source_dpi,
    )
    population_image = crop_pdf_points(
        population_page,
        tuple(args.population_crop),
        args.source_dpi,
    )
    population_image.save(args.output_dir / "ref_pop.png")
    return single_panels, population_image


def set_manuscript_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelweight": "bold",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def longest_true_run(values):
    values = np.asarray(values, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    if starts.size == 0:
        return 0, 0, 0
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(starts[best]), int(ends[best]), int(lengths[best])


def detect_single_plot_box(image, masks):
    pixels = np.asarray(image.convert("RGB"))
    flat = pixels.reshape(-1, 3)
    neutral = (
        (flat.min(axis=1) >= 180)
        & (flat.max(axis=1) < 250)
        & ((flat.max(axis=1) - flat.min(axis=1)) <= 8)
    )
    colors, counts = np.unique(flat[neutral], axis=0, return_counts=True)
    if colors.size == 0:
        raise RuntimeError("could not detect the shaded sample interval")
    shade_color = colors[int(np.argmax(counts))]
    shade = np.all(pixels == shade_color, axis=2)
    ys, xs = np.nonzero(shade)
    if xs.size == 0:
        raise RuntimeError("could not locate the shaded sample interval")

    shade_x0 = int(xs.min())
    shade_x1 = int(xs.max()) + 1
    shade_y0 = int(ys.min())
    shade_y1 = int(ys.max()) + 1
    shade_width = shade_x1 - shade_x0
    sample_duration = float(masks["sample_end"] - masks["sample_start"])
    if sample_duration <= 0 or shade_width <= 0:
        raise RuntimeError("invalid sample interval while measuring source geometry")
    pixels_per_ms = shade_width / sample_duration
    x0 = shade_x0 - masks["sample_start"] * pixels_per_ms
    x1 = x0 + masks["plot_end"] * pixels_per_ms
    return (
        float(np.clip(x0, 0, image.width)),
        float(shade_y0),
        float(np.clip(x1, 0, image.width)),
        float(shade_y1),
    )


def detect_population_plot_boxes(image):
    grayscale = np.asarray(image.convert("L"))
    dark = grayscale < 80
    height, width = dark.shape
    row_edges = [0, height // 2, height]
    column_edges = [0, width // 2, width]
    boxes = {}
    for panel_index, panel in enumerate("ABCD"):
        row, column = divmod(panel_index, 2)
        y_offset, y_end = row_edges[row], row_edges[row + 1]
        x_offset, x_end = column_edges[column], column_edges[column + 1]
        region = dark[y_offset:y_end, x_offset:x_end]

        horizontal = np.asarray([longest_true_run(line) for line in region])
        vertical = np.asarray([longest_true_run(line) for line in region.T])
        horizontal_best = horizontal[:, 2].max()
        vertical_best = vertical[:, 2].max()
        horizontal_rows = np.flatnonzero(horizontal[:, 2] >= 0.98 * horizontal_best)
        vertical_columns = np.flatnonzero(vertical[:, 2] >= 0.98 * vertical_best)
        if horizontal_rows.size == 0 or vertical_columns.size == 0:
            raise RuntimeError(f"could not detect source population panel {panel}")

        x0 = float(np.median(vertical_columns)) + x_offset
        y0 = float(np.median(vertical[vertical_columns, 0])) + y_offset
        x1 = float(np.median(horizontal[horizontal_rows, 1])) + x_offset
        y1 = float(np.median(horizontal_rows)) + y_offset
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(f"invalid detected source population panel {panel}")
        boxes[panel] = (x0, y0, x1, y1)
    return boxes


def image_pixel_box_to_figure(source_ax, image, pixel_box):
    image_position = source_ax.get_position()
    x0, y0, x1, y1 = pixel_box
    left = image_position.x0 + image_position.width * x0 / image.width
    bottom = image_position.y1 - image_position.height * y1 / image.height
    width = image_position.width * (x1 - x0) / image.width
    height = image_position.height * (y1 - y0) / image.height
    return left, bottom, width, height


def add_axis_matching_source(figure, placeholder, source_box):
    cell = placeholder.get_position()
    _, _, width, height = source_box
    if width > cell.width + 1e-9 or height > cell.height + 1e-9:
        raise RuntimeError(
            "source plot does not fit in its model-side layout cell: "
            f"source={width:.4f}x{height:.4f}, cell={cell.width:.4f}x{cell.height:.4f}"
        )
    left = cell.x0 + (cell.width - width) / 2
    bottom = cell.y0 + (cell.height - height) / 2
    placeholder.remove()
    return figure.add_axes((left, bottom, width, height))


def verify_axes_match(figure, source_boxes, model_axes):
    geometry = {}
    for panel, model_ax in model_axes.items():
        model_box = model_ax.get_position()
        _, _, source_width, source_height = source_boxes[panel]
        width_ratio = model_box.width / source_width
        height_ratio = model_box.height / source_height
        if not np.isclose(width_ratio, 1.0, rtol=0, atol=1e-10):
            raise RuntimeError(f"model panel {panel} width does not match its source")
        if not np.isclose(height_ratio, 1.0, rtol=0, atol=1e-10):
            raise RuntimeError(f"model panel {panel} height does not match its source")
        geometry[panel] = {
            "source_width_inches": source_width * figure.get_figwidth(),
            "source_height_inches": source_height * figure.get_figheight(),
            "model_width_inches": model_box.width * figure.get_figwidth(),
            "model_height_inches": model_box.height * figure.get_figheight(),
            "width_ratio": width_ratio,
            "height_ratio": height_ratio,
        }
    return geometry


def phase_profile(summary, unit, mask):
    return summary["mean_time"][:, mask, unit].mean(axis=1)


def plot_model_unit_panel(
    ax,
    panel,
    unit,
    summaries,
    masks,
    numerosity_config,
    analysis_phase,
):
    standard = summaries["standard"]
    numerosities = np.arange(
        int(numerosity_config["min_numerosity"]),
        int(numerosity_config["max_numerosity"]) + 1,
    )
    times = masks["times"]
    plot_mask = masks["plot_mask"]
    for num_index, numerosity in enumerate(numerosities):
        ax.plot(
            times[plot_mask],
            standard["mean_time"][num_index, plot_mask, unit],
            color=NUMBER_COLORS[num_index],
            linewidth=1.05,
            label=str(int(numerosity)),
        )
    ax.axvspan(
        masks["sample_start"],
        masks["sample_end"],
        color="#d9dde3",
        linewidth=0,
        zorder=-2,
    )
    ax.set_xlim(0, masks["plot_end"])
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xlabel("Time (ms)", fontsize=7, labelpad=1)
    ax.set_ylabel("Activity (a.u.)", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6, pad=1)
    ax.text(
        -0.14,
        1.03,
        panel,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        0.98,
        1.02,
        f"neuron #{unit}",
        transform=ax.transAxes,
        fontsize=6.5,
        ha="right",
        va="bottom",
    )
    ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        facecolor="white",
        framealpha=1.0,
        fontsize=5.2,
        loc="upper left",
        borderpad=0.25,
        labelspacing=0.12,
        handlelength=1.2,
        handletextpad=0.35,
    )

    inset = inset_axes(
        ax,
        width="43%",
        height="39%",
        loc="upper right",
        borderpad=0.35,
    )
    mean_key = f"{analysis_phase}_mean_activity"
    sem_key = f"{analysis_phase}_sem_activity"
    controls = PANEL_SPECS[panel]["controls"]
    for mode_index, mode in enumerate(controls):
        style = CONTROL_STYLES[mode_index % len(CONTROL_STYLES)]
        values = summaries[mode][mean_key][:, unit]
        errors = summaries[mode][sem_key][:, unit]
        inset.errorbar(
            numerosities,
            values,
            yerr=errors,
            color=style["color"],
            marker="o",
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=0.55,
            markersize=2.0,
            linewidth=0.75,
            capsize=1.0,
            label=MODE_LABELS[mode],
        )
    inset.set_xticks(numerosities)
    inset.set_xlabel("Number of items", fontsize=4.6, labelpad=0.5)
    inset.set_ylabel("Activity (a.u.)", fontsize=4.6, labelpad=0.5)
    inset.tick_params(labelsize=4.2, width=0.55, length=1.8, pad=0.8)
    inset.spines["left"].set_linewidth(0.6)
    inset.spines["bottom"].set_linewidth(0.6)
    inset.legend(
        frameon=False,
        fontsize=4.1,
        loc="best",
        borderpad=0,
        labelspacing=0.1,
        handlelength=1.0,
        handletextpad=0.25,
    )


def save_figure(fig, png_path, pdf_path, dpi):
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_single_unit_comparison(
    reference_panels,
    units,
    summaries,
    masks,
    numerosity_config,
    png_path,
    pdf_path,
    dpi,
    analysis_phase,
):
    set_manuscript_style()
    fig = plt.figure(figsize=(13.2, 8.4))
    outer = fig.add_gridspec(
        1,
        2,
        left=0.025,
        right=0.985,
        bottom=0.045,
        top=0.94,
        width_ratios=[1.0, 1.06],
        wspace=0.06,
    )
    source_grid = outer[0].subgridspec(3, 2, wspace=0.015, hspace=0.035)
    model_grid = outer[1].subgridspec(3, 2, wspace=0.32, hspace=0.26)
    panels = list("BCDEF")
    source_axes = {}
    model_placeholders = {}

    for index, panel in enumerate(panels):
        row, column = divmod(index, 2)
        source_ax = fig.add_subplot(source_grid[row, column])
        source_ax.imshow(reference_panels[panel], interpolation="lanczos")
        source_ax.axis("off")
        source_axes[panel] = source_ax
        placeholder = fig.add_subplot(model_grid[row, column])
        placeholder.axis("off")
        model_placeholders[panel] = placeholder

    fig.add_subplot(source_grid[2, 1]).axis("off")
    fig.add_subplot(model_grid[2, 1]).axis("off")
    fig.text(
        0.25,
        0.975,
        "Nieder et al. (2002)",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.75,
        0.975,
        f"Predictive RNN ({analysis_phase} period)",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    fig.canvas.draw()

    source_boxes = {
        panel: image_pixel_box_to_figure(
            source_axes[panel],
            reference_panels[panel],
            detect_single_plot_box(reference_panels[panel], masks),
        )
        for panel in panels
    }
    model_axes = {}
    for index, panel in enumerate(panels):
        model_ax = add_axis_matching_source(
            fig,
            model_placeholders[panel],
            source_boxes[panel],
        )
        plot_model_unit_panel(
            model_ax,
            panel,
            units[index],
            summaries,
            masks,
            numerosity_config,
            analysis_phase,
        )
        model_axes[panel] = model_ax
    fig.canvas.draw()
    geometry = verify_axes_match(fig, source_boxes, model_axes)
    save_figure(fig, png_path, pdf_path, dpi)
    return geometry


def plot_model_population_block(
    axes,
    sample_phase,
    delay_phase,
    numerosities,
):
    population_plots.plot_distance_panel(axes[0, 0], sample_phase, title="Sample period")
    population_plots.plot_distance_panel(axes[0, 1], delay_phase, title="Delay period")
    population_plots.plot_group_tuning_panel(axes[1, 0], sample_phase, numerosities)
    population_plots.plot_group_tuning_panel(axes[1, 1], delay_phase, numerosities)
    for letter, ax in zip("ABCD", axes.flat):
        ax.text(
            -0.18,
            1.06,
            letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )
        ax.title.set_fontsize(11)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.tick_params(labelsize=8, width=0.9, length=2.8)
        for text in ax.texts:
            if text.get_text() in {"*", "**", "***"}:
                text.set_fontsize(6.5)
        for line in ax.lines:
            line.set_linewidth(min(line.get_linewidth(), 1.5))
            if line.get_marker() not in {None, "None", ""}:
                line.set_markersize(min(line.get_markersize(), 3.5))
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(0.9)
    return axes


def plot_population_comparison(
    reference_image,
    sample_phase,
    delay_phase,
    numerosities,
    png_path,
    pdf_path,
    dpi,
):
    set_manuscript_style()
    fig = plt.figure(figsize=(13.8, 5.3))
    outer = fig.add_gridspec(
        1,
        2,
        left=0.025,
        right=0.985,
        bottom=0.075,
        top=0.91,
        width_ratios=[1.0, 1.18],
        wspace=0.06,
    )
    source_ax = fig.add_subplot(outer[0])
    source_ax.imshow(reference_image, interpolation="lanczos")
    source_ax.axis("off")
    model_grid = outer[1].subgridspec(2, 2, wspace=0.14, hspace=0.21)
    model_placeholders = {}
    for panel_index, panel in enumerate("ABCD"):
        row, column = divmod(panel_index, 2)
        placeholder = fig.add_subplot(model_grid[row, column])
        placeholder.axis("off")
        model_placeholders[panel] = placeholder
    fig.text(
        0.25,
        0.965,
        "Nieder et al. (2002)",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.75,
        0.965,
        "Predictive RNN",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    fig.canvas.draw()

    source_boxes = {
        panel: image_pixel_box_to_figure(source_ax, reference_image, pixel_box)
        for panel, pixel_box in detect_population_plot_boxes(reference_image).items()
    }
    model_axes_by_panel = {
        panel: add_axis_matching_source(
            fig,
            model_placeholders[panel],
            source_boxes[panel],
        )
        for panel in "ABCD"
    }
    model_axes = np.asarray([model_axes_by_panel[panel] for panel in "ABCD"]).reshape(2, 2)
    plot_model_population_block(
        model_axes,
        sample_phase,
        delay_phase,
        numerosities,
    )
    fig.canvas.draw()
    geometry = verify_axes_match(fig, source_boxes, model_axes_by_panel)
    save_figure(fig, png_path, pdf_path, dpi)
    return geometry


def main():
    args = parse_args()
    validate_crop(tuple(args.population_crop), "population")
    single_crops = parse_crop_overrides(args.single_panel_crop)
    if not args.paper_pdf.exists():
        raise FileNotFoundError(f"reference PDF not found: {args.paper_pdf}")
    if args.trials_per_numerosity < 2:
        raise ValueError("trials-per-numerosity must be at least 2")

    args.output_dir = output_directory(args.output_dir)
    cache_path = args.analysis_cache or args.output_dir / "cache.npz"
    device = cells.resolve_device(args.device)
    config, sim_config, numerosity_config, metadata = cells.load_config(args.config)
    checkpoint = max(list_checkpoints(args), key=parse_epoch)
    epoch = parse_epoch(checkpoint)
    masks = cells.get_time_masks(sim_config, numerosity_config, metadata)
    analysis, fingerprint, cache_hit = load_or_compute_analysis(
        args,
        config,
        sim_config,
        numerosity_config,
        metadata,
        masks,
        checkpoint,
        device,
        cache_path,
    )

    n_units = analysis["summaries"]["standard"]["mean_time"].shape[-1]
    for phase, units in (
        ("sample", args.sample_units),
        ("delay", args.delay_units),
    ):
        for panel, unit in zip("BCDEF", units):
            if unit < 0 or unit >= n_units:
                raise ValueError(f"{phase} panel {panel} unit {unit} is outside [0, {n_units - 1}]")

    reference_panels, reference_population = extract_reference_panels(args, single_crops)
    numerosities = np.arange(
        int(numerosity_config["min_numerosity"]),
        int(numerosity_config["max_numerosity"]) + 1,
    )
    sample_phase = population.phase_population(
        analysis["sample_features"],
        analysis["labels"],
        numerosities,
        args.alpha,
        args.cell_threshold,
    )
    delay_phase = population.phase_population(
        analysis["delay_features"],
        analysis["labels"],
        numerosities,
        args.alpha,
        args.cell_threshold,
    )

    sample_single_png = args.output_dir / "sample.png"
    sample_single_pdf = args.output_dir / "sample.pdf"
    delay_single_png = args.output_dir / "delay.png"
    delay_single_pdf = args.output_dir / "delay.pdf"
    population_png = args.output_dir / "population.png"
    population_pdf = args.output_dir / "population.pdf"
    sample_geometry = plot_single_unit_comparison(
        reference_panels,
        args.sample_units,
        analysis["summaries"],
        masks,
        numerosity_config,
        sample_single_png,
        sample_single_pdf,
        args.dpi,
        "sample",
    )
    delay_geometry = plot_single_unit_comparison(
        reference_panels,
        args.delay_units,
        analysis["summaries"],
        masks,
        numerosity_config,
        delay_single_png,
        delay_single_pdf,
        args.dpi,
        "delay",
    )
    population_geometry = plot_population_comparison(
        reference_population,
        sample_phase,
        delay_phase,
        numerosities,
        population_png,
        population_pdf,
        args.dpi,
    )

    manifest = {
        "schema_version": 2,
        "citation": PAPER_CITATION,
        "doi": PAPER_DOI,
        "reference_pdf": str(args.paper_pdf),
        "reference_pdf_sha256": file_sha256(args.paper_pdf),
        "reference_pages": {
            "single_units": args.paper_single_page,
            "population": args.paper_population_page,
        },
        "reference_crops_pdf_points_top_left": {
            "single_units": {panel: list(box) for panel, box in single_crops.items()},
            "population": list(args.population_crop),
        },
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "epoch": epoch,
        "analysis_fingerprint": fingerprint,
        "analysis_cache": str(cache_path),
        "analysis_cache_hit": cache_hit,
        "device": str(device),
        "trials_per_numerosity": args.trials_per_numerosity,
        "seed": args.seed,
        "units": {
            "sample": {panel: unit for panel, unit in zip("BCDEF", args.sample_units)},
            "delay": {panel: unit for panel, unit in zip("BCDEF", args.delay_units)},
        },
        "population": {
            "alpha": args.alpha,
            "cell_threshold": args.cell_threshold,
            "sample_selective_units": int(len(sample_phase["units"])),
            "delay_selective_units": int(len(delay_phase["units"])),
        },
        "plot_geometry": {
            "description": (
                "Model data-axis rectangles are matched to the corresponding "
                "source data-axis rectangles before rendering."
            ),
            "single_units_sample": sample_geometry,
            "single_units_delay": delay_geometry,
            "population": population_geometry,
        },
        "outputs": {
            "single_units_sample_png": str(sample_single_png),
            "single_units_sample_pdf": str(sample_single_pdf),
            "single_units_delay_png": str(delay_single_png),
            "single_units_delay_pdf": str(delay_single_pdf),
            "population_png": str(population_png),
            "population_pdf": str(population_pdf),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
