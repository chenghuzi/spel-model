"""Run manifests and short checkpoint paths shared by training and analysis."""

import glob
import hashlib
import json
import re
import time
from pathlib import Path

from paths import ARTIFACTS, ROOT, output_directory, output_file

PUBLISHED_CHECKPOINT = ARTIFACTS / "numerosity" / "published" / "ep000150.pt"


def phase_name(trial_type):
    return {"prepotent": "pre", "balanced": "bal", "numerosity": "num"}[trial_type]


def checkpoint_path(directory, trial_type, epoch):
    return output_file(directory, f"{phase_name(trial_type)}-ep{epoch:06d}.pt")


def run_directory(config, command, run_id=None):
    # Record the full experiment identity once; keep filenames independent of config size.
    identity = {"command": command, "config": config}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    if run_id is None:
        task = "num" if config["task"] == "numerosity" else "dpx"
        run_id = f"{task}-{hashlib.sha256(encoded.encode()).hexdigest()[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,19}", run_id):
        raise ValueError("Run IDs must contain 1-20 letters, digits, underscores, or hyphens")
    directory = output_directory(ARTIFACTS / "runs" / run_id)
    manifest = directory / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError(f"Run ID already belongs to another configuration: {run_id}")
    manifest.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return directory


def wandb_directory():
    # W&B controls its own deep, long-named SDK cache; keep it outside the project tree.
    directory = Path.home() / ".cache" / "spiking-spel" / "wandb"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def parse_epoch(path):
    match = re.search(r"(?:^|-)ep(\d+)\.pt$", Path(path).name)
    if match is None:
        raise ValueError(f"Expected an epoch checkpoint, got {path}")
    return int(match.group(1))


def list_checkpoints(args):
    # Use explicit paths or a glob; the CLI default points at the preserved manuscript model.
    if args.checkpoint_glob is None:
        candidates = [args.checkpoint]
    else:
        pattern = Path(args.checkpoint_glob)
        if not pattern.is_absolute():
            pattern = ROOT / pattern
        candidates = [Path(path) for path in glob.glob(str(pattern))]
    now = time.time()
    checkpoints = [
        path for path in candidates if now - path.stat().st_mtime >= args.min_checkpoint_age
    ]
    checkpoints = sorted(set(checkpoints), key=parse_epoch)
    if args.epochs:
        checkpoints = [path for path in checkpoints if parse_epoch(path) in args.epochs]
    if args.max_checkpoints > 0:
        checkpoints = checkpoints[-args.max_checkpoints :]
    if not checkpoints:
        raise FileNotFoundError("No checkpoint matches the requested path, epochs, and age")
    return checkpoints
