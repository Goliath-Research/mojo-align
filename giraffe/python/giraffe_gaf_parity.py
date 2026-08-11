"""Compare Mojo Giraffe GAF to a golden / vg GAF on MethylCall-critical fields."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _bare_qname(q: str) -> str:
    return q.split("_")[0]


def _cs_tag(cols: List[str]) -> str:
    for c in cols[12:]:
        if c.startswith("cs:Z:"):
            return c
    return ""


def _tag(cols: List[str], prefix: str) -> Optional[str]:
    for c in cols[12:]:
        if c.startswith(prefix):
            return c
    return None


def parse_gaf(path: str) -> Dict[Tuple[str, str], dict]:
    """Key by (bare_qname, ri) when ri present else (bare_qname, path)."""
    out: Dict[Tuple[str, str], dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("@"):
                continue
            cols = line.split("\t")
            if len(cols) < 12:
                continue
            if cols[5] == "*":
                continue
            bare = _bare_qname(cols[0])
            ri = _tag(cols, "ri:i:")
            key2 = ri if ri else cols[5]
            out[(bare, key2)] = {
                "path": cols[5],
                "mapq": cols[11],
                "cs": _cs_tag(cols),
                "os": _tag(cols, "os:Z:"),
                "rc": _tag(cols, "rc:Z:"),
                "ri": ri,
            }
    return out


def compare(mojo: dict, golden: dict, require_extra: bool) -> int:
    ok = 0
    bad = 0
    only_m = sorted(set(mojo) - set(golden))
    only_g = sorted(set(golden) - set(mojo))
    for k in only_m:
        print(f"ONLY_MOJO {k}")
        bad += 1
    for k in only_g:
        print(f"ONLY_GOLDEN {k}")
        bad += 1
    for k in sorted(set(mojo) & set(golden)):
        a, b = mojo[k], golden[k]
        mismatches = []
        if a["path"] != b["path"]:
            mismatches.append(f"path {a['path']}!={b['path']}")
        if a["cs"] != b["cs"]:
            mismatches.append(f"cs {a['cs']}!={b['cs']}")
        if require_extra:
            for t in ("os", "rc", "ri"):
                if a.get(t) and b.get(t) and a[t] != b[t]:
                    mismatches.append(f"{t} {a[t]}!={b[t]}")
        if mismatches:
            print(f"MISMATCH {k}: {'; '.join(mismatches)}")
            bad += 1
        else:
            ok += 1
    print(f"parity_ok={ok} parity_bad={bad}")
    return 0 if bad == 0 and ok > 0 else 1
