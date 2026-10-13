import argparse
import json
import os
from pathlib import Path

# Keep analysis from silently turning into another GPU job.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


import numpy as np
import torch
from scipy.stats import f_oneway

from environments.stimuli import NUMEROSITY_STIMULUS_MODES, DotArrayGenerator
from models import build_model
from models.config import get_model_metadata, read_config
from models.storage import PUBLISHED_CHECKPOINT, list_checkpoints, parse_epoch
from paths import ARTIFACTS, ROOT, output_directory


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure numerosity selectivity and held-out decoding for manuscript checkpoints."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/numerosity.json")
    parser.add_argument("--checkpoint", type=Path, default=PUBLISHED_CHECKPOINT)
    parser.add_argument("--checkpoint-glob", default=None)
    parser.add_argument("--epochs", nargs="*", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--min-checkpoint-age", type=float, default=20.0)
    parser.add_argument("--trials-per-numerosity", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--stimulus-modes",
        nargs="+",
        choices=NUMEROSITY_STIMULUS_MODES,
        default=[
            "standard",
            "equal_area",
            "equal_circumference",
            "shape",
            "linear",
            "high_density",
            "low_density",
            "variable_features",
        ],
    )
    parser.add_argument(
        "--analysis-phase",
        choices=["sample", "delay", "test"],
        default="delay",
        help="Trial phase used for decoding and tuning statistics.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS / "numerosity" / "selectivity",
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def load_config(config_path):
    # Training and analysis use the same model/task preparation contract.
    config = read_config(config_path)
    sim_config = config["sim_config"]
    numerosity_config = config["numerosity_config"]
    metadata = get_model_metadata(config)
    return config, sim_config, numerosity_config, metadata


def get_time_masks(sim_config, numerosity_config, metadata):
    dt = sim_config["dt"]
    n_steps = int(metadata["trial_end_ms"] / dt)
    times = np.arange(n_steps) * dt
    sample_start = numerosity_config["fixation_ms"]
    sample_end = sample_start + numerosity_config["sample_ms"]
    test_start = sample_end + numerosity_config["delay_ms"]
    test_end = test_start + numerosity_config["test_ms"]
    sample_mask = (times > sample_start) & (times <= sample_end)
    delay_mask = (times > sample_end) & (times <= test_start)
    test_mask = (times > test_start) & (times <= test_end)
    # Nieder et al. Figure 2 stops at test onset and displays fixation,
    # sample, and memory-delay activity only.
    plot_end = test_start
    plot_mask = times <= plot_end
    return {
        "times": times,
        "sample_mask": sample_mask,
        "delay_mask": delay_mask,
        "test_mask": test_mask,
        "plot_mask": plot_mask,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "test_start": test_start,
        "test_end": test_end,
        "plot_end": plot_end,
    }


def get_analysis_mask(masks, analysis_phase):
    return masks[f"{analysis_phase}_mask"]


def build_match_sequences(
    numerosity_config,
    metadata,
    masks,
    stimulus_mode,
    trials_per_numerosity,
    seed,
):
    np_state = np.random.get_state()
    np.random.seed(seed)
    try:
        generator = DotArrayGenerator(
            grid_size=numerosity_config["grid_size"],
            stimulus_mode=stimulus_mode,
            min_dot_distance=numerosity_config["min_dot_distance"],
            stimulus_family=numerosity_config["stimulus_family"],
            position_lattice_size=numerosity_config["position_lattice_size"],
            aperture_radius=numerosity_config["aperture_radius"],
            aperture_value=numerosity_config["aperture_value"],
            dot_value=numerosity_config["dot_value"],
            dot_radius_values=numerosity_config["dot_radius_values"],
            equal_area_total_area=numerosity_config["equal_area_total_area"],
            density_candidate_samples=numerosity_config["density_candidate_samples"],
        )
        min_n = int(numerosity_config["min_numerosity"])
        max_n = int(numerosity_config["max_numerosity"])
        n_steps = len(masks["times"])
        n_in = metadata["n_in"]
        sequences = []
        labels = []
        for numerosity in range(min_n, max_n + 1):
            for _ in range(trials_per_numerosity):
                linear_angle = (
                    float(np.random.uniform(0.0, np.pi)) if stimulus_mode == "linear" else None
                )
                sample_stimulus = generator.render(
                    numerosity,
                    linear_angle=linear_angle,
                ).astype(np.float32)
                test_stimulus = generator.render(
                    numerosity,
                    linear_angle=linear_angle,
                ).astype(np.float32)
                sequence = np.zeros((n_steps, n_in), dtype=np.float32)
                sequence[masks["sample_mask"]] = sample_stimulus
                sequence[masks["test_mask"]] = test_stimulus
                sequences.append(sequence)
                labels.append(numerosity - min_n)
    finally:
        np.random.set_state(np_state)
    return torch.from_numpy(np.stack(sequences)).float(), np.array(labels, dtype=np.int64)


def build_model_for_checkpoint(config, sim_config, metadata, device):
    model = build_model(
        metadata["n_in"],
        int(config["n_hidden"]),
        metadata["n_o"],
        sim_config,
    )
    model.to(device)
    model.eval()
    return model


def run_model(model, sequences, batch_size, device, seed):
    hiddens = []
    outputs = []
    torch.manual_seed(seed)
    with torch.no_grad():
        for start in range(0, sequences.shape[0], batch_size):
            batch = sequences[start : start + batch_size].to(device)
            _, hidden, output, _ = model(batch)
            hiddens.append(hidden.cpu().numpy())
            outputs.append(output.cpu().numpy())
    return np.concatenate(hiddens, axis=0), np.concatenate(outputs, axis=0)


def fdr_bh(p_values):
    p_values = np.asarray(p_values, dtype=float)
    # NaN denotes an undefined test; measured p-values must be probabilities.
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("FDR requires p-values in [0, 1] or NaN for undefined tests")
    # Undefined tests remain undefined; only measured p-values participate in FDR.
    adjusted = np.full_like(p_values, np.nan)
    finite = np.isfinite(p_values)
    if not finite.any():
        return adjusted

    finite_p = p_values[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    n = len(ranked)
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    finite_adjusted = np.empty_like(finite_p)
    finite_adjusted[order] = q_ranked
    adjusted[finite] = finite_adjusted
    return adjusted


def analyze_units(features, labels, alpha):
    # Reject invalid observations at the analysis input boundary, before statistical testing.
    if not np.isfinite(features).all():
        raise ValueError("Unit analysis requires finite activity values")
    classes = np.unique(labels)
    groups = [features[labels == label] for label in classes]
    responses = np.stack([group.mean(axis=0) for group in groups])
    sem = np.stack([group.std(axis=0) / np.sqrt(group.shape[0]) for group in groups])

    f_scores = []
    p_values = []
    for unit_idx in range(features.shape[1]):
        # Preserve SciPy's undefined NaN and valid infinite F statistics.
        f_score, p_value = f_oneway(*[group[:, unit_idx] for group in groups])
        f_scores.append(float(f_score))
        p_values.append(float(p_value))

    f_scores = np.array(f_scores)
    p_values = np.array(p_values)
    q_values = fdr_bh(p_values)
    preferred = np.argmax(responses, axis=0)
    return {
        "responses": responses,
        "sem": sem,
        "f_scores": f_scores,
        "p_unc": p_values,
        "p_fdr": q_values,
        "preferred": preferred,
        "tuned_uncorrected": np.where(p_values < alpha)[0],
        "tuned_fdr": np.where(q_values < alpha)[0],
    }


def ridge_decode(features, labels, seed, repeats=8, ridge_lambda=1e-2):
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    accuracies = []
    per_class_accs = []
    for _ in range(repeats):
        train_indices = []
        test_indices = []
        for label in classes:
            label_indices = np.where(labels == label)[0]
            shuffled = rng.permutation(label_indices)
            split = int(0.7 * len(shuffled))
            train_indices.extend(shuffled[:split])
            test_indices.extend(shuffled[split:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        x_train = features[train_indices]
        x_test = features[test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]

        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
        x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
        y_onehot = np.eye(len(classes))[y_train]

        regularizer = ridge_lambda * np.eye(x_train.shape[1])
        regularizer[-1, -1] = 0.0
        weights = np.linalg.solve(
            x_train.T @ x_train + regularizer,
            x_train.T @ y_onehot,
        )
        predictions = np.argmax(x_test @ weights, axis=1)
        accuracies.append(float(np.mean(predictions == y_test)))
        per_class_accs.append(
            [float(np.mean(predictions[y_test == label] == label)) for label in classes]
        )

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "per_class_accuracy_mean": [
            float(value) for value in np.mean(np.array(per_class_accs), axis=0)
        ],
        "chance": float(1.0 / len(classes)),
    }


def summarize_timecourses(hiddens, labels, masks, analysis_phase):
    classes = np.unique(labels)
    mean_time = np.stack([hiddens[labels == label].mean(axis=0) for label in classes])
    sem_time = np.stack(
        [
            hiddens[labels == label].std(axis=0) / np.sqrt(np.sum(labels == label))
            for label in classes
        ]
    )
    analysis_mask = get_analysis_mask(masks, analysis_phase)
    phase_values = np.stack(
        [hiddens[labels == label][:, analysis_mask, :].mean(axis=1) for label in classes]
    )
    mean_activity = phase_values.mean(axis=1)
    sem_activity = phase_values.std(axis=1) / np.sqrt(phase_values.shape[1])
    return mean_time, sem_time, mean_activity, sem_activity


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def main():
    args = parse_args()
    device = resolve_device(args.device)
    config, sim_config, numerosity_config, metadata = load_config(args.config)
    checkpoints = list_checkpoints(args)
    masks = get_time_masks(sim_config, numerosity_config, metadata)
    analysis_mask = get_analysis_mask(masks, args.analysis_phase)
    args.output_dir = output_directory(args.output_dir)

    sequence_cache = {}
    for mode_index, mode in enumerate(args.stimulus_modes):
        sequence_cache[mode] = build_match_sequences(
            numerosity_config,
            metadata,
            masks,
            mode,
            args.trials_per_numerosity,
            seed=args.seed + mode_index * 1000,
        )

    results = []
    for checkpoint in checkpoints:
        epoch = parse_epoch(checkpoint)
        print(f"analyzing epoch {epoch}: {checkpoint}")
        model = build_model_for_checkpoint(config, sim_config, metadata, device)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)

        mode_results = {}
        for mode_index, mode in enumerate(args.stimulus_modes):
            sequences, labels = sequence_cache[mode]
            hiddens, outputs = run_model(
                model,
                sequences,
                args.batch_size,
                device,
                seed=args.seed + epoch * 17 + mode_index,
            )
            phase_features = hiddens[:, analysis_mask, :].mean(axis=1)
            unit_stats = analyze_units(phase_features, labels, args.alpha)
            decode = ridge_decode(phase_features, labels, seed=args.seed + epoch)
            mode_results[mode] = {
                "decode": decode,
                "tuned_uncorrected_count": int(len(unit_stats["tuned_uncorrected"])),
                "tuned_fdr_count": int(len(unit_stats["tuned_fdr"])),
                "preferred_counts_uncorrected": np.bincount(
                    unit_stats["preferred"][unit_stats["tuned_uncorrected"]],
                    minlength=decode["per_class_accuracy_mean"].__len__(),
                ),
                "preferred_counts_fdr": np.bincount(
                    unit_stats["preferred"][unit_stats["tuned_fdr"]],
                    minlength=decode["per_class_accuracy_mean"].__len__(),
                ),
                "min_p_uncorrected": float(unit_stats["p_unc"].min()),
                "min_p_fdr": float(unit_stats["p_fdr"].min()),
                "median_p_uncorrected": float(np.median(unit_stats["p_unc"])),
                "median_p_fdr": float(np.median(unit_stats["p_fdr"])),
            }
        results.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "modes": mode_results,
            }
        )

    # Save the per-checkpoint selectivity and held-out decoding results from Methods.
    json_path = args.output_dir / "selectivity.json"
    payload = {
        "config": str(args.config),
        "device": str(device),
        "alpha": args.alpha,
        "analysis_phase": args.analysis_phase,
        "trials_per_numerosity": args.trials_per_numerosity,
        "batch_size": args.batch_size,
        "stimulus_modes": args.stimulus_modes,
        "seed": args.seed,
        "results": results,
    }
    json_path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    print(json_path)


if __name__ == "__main__":
    main()
