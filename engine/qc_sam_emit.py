"""Fast batch SAM formatting for MojoGiraffe QC (called from Mojo via Python).

Canonical implementation lives in ``engine.grch38_offsets`` (overlay-mounted on
sisters). This module re-exports that API for older call sites.
"""

from __future__ import annotations

from engine.grch38_offsets import (  # noqa: F401
    append_hits,
    close_sam,
    open_sam,
    summary,
)
