"""Configuration helpers shared by DeTurb command-line entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .paths import explicit_destinations, normalize_paths, resolve_path


def load_config_section(config_path: str | Path, section: str) -> dict[str, Any]:
    path = Path(resolve_path(config_path))
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    values = config.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"Config section {section!r} must be a JSON object")
    return values


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    section: str,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Load defaults from a JSON section, then let CLI arguments override them."""

    tokens = list(sys.argv[1:] if argv is None else argv)
    known_args, _ = parser.parse_known_args(tokens)
    config_path = getattr(known_args, "config", None)
    defaults = {}
    if config_path:
        config_path = resolve_path(config_path)
        defaults = load_config_section(config_path, section)
        valid_destinations = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(defaults) - valid_destinations)
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in config section {section!r}: "
                + ", ".join(unknown_keys)
            )
        parser.set_defaults(**defaults)

    args = parser.parse_args(tokens)
    inherited = set(defaults) - explicit_destinations(parser, tokens)
    try:
        return normalize_paths(args, config_base=Path(config_path).parent if config_path else None,
                               inherited=inherited)
    except ValueError as error:
        parser.error(str(error))
