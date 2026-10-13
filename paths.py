"""Project paths and the shared short-name policy for generated files."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def output_directory(path):
    # Validate user-selected destinations before writers create any project output.
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        parts = path.relative_to(ROOT).parts
        if len(parts) > 3 or any(len(part) > 20 for part in parts):
            raise ValueError(
                "Output directories must leave room for a file within four short path components"
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_file(directory, filename):
    # Check dynamic filenames, including checkpoint epochs, at the filesystem boundary.
    if len(filename) > 20 or Path(filename).name != filename:
        raise ValueError(f"Expected a filename of at most 20 characters: {filename!r}")
    return output_directory(directory) / filename


def file_sha256(path):
    # Stream large checkpoints and recordings rather than duplicating them in memory.
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_file(parameters, filename):
    # Store cache provenance next to a stable short identity instead of in its filename.
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    identity = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    directory = output_directory(ARTIFACTS / "cache" / identity)
    manifest = directory / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != parameters:
        raise ValueError(f"Cache identity collision: {directory}")
    manifest.write_text(json.dumps(parameters, indent=2, sort_keys=True) + "\n")
    return output_file(directory, filename)
