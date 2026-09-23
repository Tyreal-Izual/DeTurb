"""Path resolution shared by installed command-line entrypoints.

CLI paths are relative to the caller; paths inherited from JSON are relative
to that JSON file. Environment expansion is limited to known path arguments.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Sequence

_ENV = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
_PATH_FIELDS = {
    "config", "input", "output", "manifest", "checkpoint", "load",
    "experiment", "comparison", "phase1_run", "resumed_run", "continuous_run",
    "pre_edge_experiment", "edge_experiment", "train_manifest", "test_manifest",
}


def is_path_argument(name: str) -> bool:
    return name in _PATH_FIELDS or name.endswith(("_path", "_root", "_checkpoint"))


def resolve_path(value: str | Path, base: str | Path | None = None) -> str:
    """Expand a path without creating it; undefined variables are errors."""
    def substitute(match: re.Match) -> str:
        name = match.group(1) or match.group(2)
        if name not in os.environ or not os.environ[name]:
            raise ValueError(f"Path requires environment variable {name}")
        return os.environ[name]

    path = Path(_ENV.sub(substitute, os.fspath(value))).expanduser()
    if not path.is_absolute():
        path = (Path(base) if base is not None else Path.cwd()) / path
    return str(path.resolve())


def explicit_destinations(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    options = {option: action.dest for action in parser._actions for option in action.option_strings}
    result = set()
    for token in argv:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        if option in options:
            result.add(options[option])
        elif option.startswith("--") and parser.allow_abbrev:
            matches = [dest for name, dest in options.items() if name.startswith(option)]
            if len(matches) == 1:
                result.add(matches[0])
    return result


def normalize_paths(args: argparse.Namespace, *, config_base: Path | None = None,
                    inherited: set[str] | None = None) -> argparse.Namespace:
    for name, value in vars(args).items():
        if not is_path_argument(name) or value is None or value == "":
            continue
        if name == "load" and value == "latest":
            continue
        base = config_base if inherited and name in inherited else None
        setattr(args, name, resolve_path(value, base))
    return args


def parse_path_args(parser: argparse.ArgumentParser,
                    argv: Sequence[str] | None = None) -> argparse.Namespace:
    try:
        return normalize_paths(parser.parse_args(argv))
    except ValueError as error:
        parser.error(str(error))
