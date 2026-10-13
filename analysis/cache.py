"""Activity collection and cache IO shared by numerosity figure entry points."""

import json

import numpy as np
import torch

from analysis import numerosity as cells
from environments.stimuli import NUMEROSITY_STIMULUS_MODES
from models.storage import parse_epoch
from paths import ROOT, file_sha256, output_file

ANALYSIS_MODES = NUMEROSITY_STIMULUS_MODES
SUMMARY_FIELDS = (
    "mean_time",
    "sem_time",
    "sample_mean_activity",
    "sample_sem_activity",
    "delay_mean_activity",
    "delay_sem_activity",
)


def analysis_fingerprint(args, checkpoint):
    # Hash all source modules that now own the stimulus, model, and analysis computations.
    sources = [
        "environments/stimuli.py",
        "environments/numerosity.py",
        "models/layers.py",
        "models/visual.py",
        "models/config.py",
        "analysis/numerosity.py",
        "analysis/cache.py",
    ]
    return {
        "schema_version": 3,
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(args.config),
        "sources": {name: file_sha256(ROOT / name) for name in sources},
        "trials_per_numerosity": args.trials_per_numerosity,
        "seed": args.seed,
        "modes": ANALYSIS_MODES,
    }


def cache_key(mode, field):
    return f"summary__{mode}__{field}"


def load_analysis_cache(path, fingerprint):
    if not path.exists():
        return None
    # Existing cache read/format errors are actionable, not ordinary cache misses.
    with np.load(path, allow_pickle=False) as cached:
        cached_fingerprint = json.loads(str(cached["fingerprint"].item()))
        if cached_fingerprint != fingerprint:
            return None
        summaries = {}
        for mode in ANALYSIS_MODES:
            summaries[mode] = {
                field: cached[cache_key(mode, field)].copy() for field in SUMMARY_FIELDS
            }
        return {
            "summaries": summaries,
            "sample_features": cached["sample_features"].copy(),
            "delay_features": cached["delay_features"].copy(),
            "labels": cached["labels"].copy(),
        }


def save_analysis_cache(path, fingerprint, analysis):
    if path.suffix != ".npz":
        raise ValueError("Analysis cache files must use the .npz extension")
    path = output_file(path.parent, path.name)
    arrays = {
        "fingerprint": np.asarray(json.dumps(fingerprint, sort_keys=True)),
        "sample_features": analysis["sample_features"],
        "delay_features": analysis["delay_features"],
        "labels": analysis["labels"],
    }
    for mode, summary in analysis["summaries"].items():
        for field, values in summary.items():
            arrays[cache_key(mode, field)] = values
    np.savez_compressed(path, **arrays)


def compute_model_analysis(
    args,
    config,
    sim_config,
    numerosity_config,
    metadata,
    masks,
    checkpoint,
    device,
):
    model = cells.build_model_for_checkpoint(config, sim_config, metadata, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    summaries = {}
    sample_features = None
    delay_features = None
    standard_labels = None
    epoch = parse_epoch(checkpoint)
    for mode_index, mode in enumerate(ANALYSIS_MODES):
        print(f"analyzing {mode} stimuli")
        sequences, labels = cells.build_match_sequences(
            numerosity_config,
            metadata,
            masks,
            mode,
            args.trials_per_numerosity,
            seed=args.seed + 1000 * mode_index,
        )
        hiddens, _ = cells.run_model(
            model,
            sequences,
            args.batch_size,
            device,
            seed=args.seed + epoch * 17 + mode_index,
        )
        mean_time, sem_time, sample_mean, sample_sem = cells.summarize_timecourses(
            hiddens,
            labels,
            masks,
            "sample",
        )
        _, _, delay_mean, delay_sem = cells.summarize_timecourses(
            hiddens,
            labels,
            masks,
            "delay",
        )
        summaries[mode] = {
            "mean_time": mean_time,
            "sem_time": sem_time,
            "sample_mean_activity": sample_mean,
            "sample_sem_activity": sample_sem,
            "delay_mean_activity": delay_mean,
            "delay_sem_activity": delay_sem,
        }
        if mode == "standard":
            sample_features = hiddens[:, masks["sample_mask"], :].mean(axis=1)
            delay_features = hiddens[:, masks["delay_mask"], :].mean(axis=1)
            standard_labels = labels.copy()

    return {
        "summaries": summaries,
        "sample_features": sample_features,
        "delay_features": delay_features,
        "labels": standard_labels,
    }
