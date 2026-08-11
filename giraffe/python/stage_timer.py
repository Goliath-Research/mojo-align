"""Lightweight stage timer for quartet_map / linear bakeoffs."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional


@dataclass
class StageTimer:
    stages: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "StageTimer":
        raw = os.environ.get("METHYLGRAPHER_PROFILE_STAGES", "1").strip().lower()
        return cls(enabled=raw not in {"0", "false", "no", "off"})

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.stages[name] = self.stages.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def add(self, name: str, seconds: float, n: int = 1) -> None:
        if not self.enabled:
            return
        self.stages[name] = self.stages.get(name, 0.0) + float(seconds)
        self.counts[name] = self.counts.get(name, 0) + n

    def report(self) -> Dict[str, object]:
        total = sum(self.stages.values()) or 1e-12
        return {
            "total_s": round(total, 6),
            "stages_s": {k: round(v, 6) for k, v in sorted(self.stages.items())},
            "stages_pct": {
                k: round(100.0 * v / total, 2) for k, v in sorted(self.stages.items())
            },
            "counts": dict(sorted(self.counts.items())),
        }

    def write(self, path: Optional[str] = None) -> str:
        payload = self.report()
        text = json.dumps(payload, indent=2) + "\n"
        out = path or os.environ.get("METHYLGRAPHER_PROFILE_JSON", "")
        if out:
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(text)
        print("STAGE_TIMER " + json.dumps(payload), flush=True)
        return text
