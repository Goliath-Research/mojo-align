"""Load a module implementation from a sibling package path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_sibling(module_attr_name: str, rel_path: str) -> ModuleType:
    """Load ``mojo-align/<rel_path>`` as ``engine.<module_attr_name>`` impl."""
    root = Path(__file__).resolve().parents[2]
    path = root / rel_path
    # Unique name avoids colliding with this shim module.
    full_name = f"engine._impl_{module_attr_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shim target: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod
