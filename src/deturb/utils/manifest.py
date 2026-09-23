"""Load sample names from an existing training/evaluation manifest."""

from __future__ import annotations

import json
from pathlib import Path


def load_manifest_names(path: str | Path, split: str) -> list[str]:
    manifest = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    try:
        split_manifest = manifest["splits"][split]
        samples = split_manifest["samples"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Manifest does not contain split {split!r}") from error
    names = [sample["name"] for sample in samples]
    if len(names) != len(set(names)):
        raise ValueError(f"Manifest split {split!r} contains duplicate filenames")
    return names
