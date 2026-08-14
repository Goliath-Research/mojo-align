"""Align map backends for methylGrapher dual-graph GAF production.

Backends (``METHYLGRAPHER_ALIGN_ENGINE`` / ``align_engine``):

- ``cpu_vg`` — stock ``vg giraffe -o gaf --named-coordinates`` (rollback).
- ``mojo_giraffe`` — Mojo Giraffe (GBZ-native when quartet present; else GFA).
- ``gpu_giraffe`` — prefers Mojo GBZ/GFA; emergency ``FALLBACK=vg``.

Environment:

- ``METHYLGRAPHER_GPU_GIRAFFE_FALLBACK`` — ``mojo`` (default) | ``vg`` | ``error``
- ``METHYLGRAPHER_MOJO_GIRAFFE_BIN`` — path to ``methylGrapher`` / wrapper
- ``METHYLGRAPHER_GIRAFFE_GFA`` — explicit GFA for fixture path
- ``METHYLGRAPHER_MOJO_GIRAFFE_MAX_GFA_BYTES`` — GFA size cap (GBZ path ignores this)
- ``METHYLGRAPHER_GIRAFFE_DEVICE`` — ``auto`` | ``cpu`` | ``nvidia`` | ``amd``
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Literal, Optional

from engine.giraffe_gbz_helper import resolve_gbz_quartet, segment_cache_ready
from engine.quartet_map import mojo_giraffe_ready

AlignEngine = Literal["cpu_vg", "gpu_giraffe", "mojo_giraffe"]

_DEFAULT_MAX_GFA_BYTES = 64 * 1024 * 1024


def normalize_align_engine(value: Optional[str]) -> AlignEngine:
    if value is None or str(value).strip() == "":
        env = os.environ.get("METHYLGRAPHER_ALIGN_ENGINE", "cpu_vg")
        value = env
    eng = str(value).strip().lower()
    if eng in {"cpu_vg", "cpu", "vg"}:
        return "cpu_vg"
    if eng in {"mojo_giraffe", "mojo"}:
        return "mojo_giraffe"
    if eng in {"gpu_giraffe", "gpu", "gh200"}:
        return "gpu_giraffe"
    raise RuntimeError(
        "align_engine must be 'cpu_vg', 'gpu_giraffe', or 'mojo_giraffe' "
        f"(got {value!r})"
    )


def gpu_giraffe_fallback() -> str:
    raw = os.environ.get("METHYLGRAPHER_GPU_GIRAFFE_FALLBACK", "mojo").strip().lower()
    if raw in {"mojo", "mojo_giraffe"}:
        return "mojo"
    if raw in {"vg", "cpu_vg", "fallback_vg"}:
        return "vg"
    if raw in {"error", "fail", "none"}:
        return "error"
    raise RuntimeError(
        "METHYLGRAPHER_GPU_GIRAFFE_FALLBACK must be 'mojo', 'vg', or 'error' "
        f"(got {raw!r})"
    )


def build_vg_giraffe_gaf_cmd(
    *,
    vg_path: str,
    thread: int,
    output_format: str,
    index_params: str,
    giraffe_input: str,
) -> str:
    if output_format.lower() != "gaf":
        raise RuntimeError(
            f"Science Align requires GAF output (got {output_format!r}); "
            "do not use BAM for MethylCall"
        )
    return (
        f"{vg_path} giraffe -p -t {thread} -o {output_format} -M 2 "
        f"--named-coordinates {index_params} {giraffe_input}"
    )


def resolve_mojo_giraffe_bin() -> Optional[str]:
    env = os.environ.get("METHYLGRAPHER_MOJO_GIRAFFE_BIN", "").strip()
    if env and Path(env).exists():
        return env
    # Host checkouts: mojo-align/bin or staged flat image /opt/.../bin
    here = Path(__file__).resolve()
    for cand in (
        here.parents[2] / "bin" / "methylGrapher",  # mojo-align/bin
        here.parents[1] / "bin" / "methylGrapher",  # flat _flat_image/bin
        Path("/opt/methylgrapher-mojo/bin/methylGrapher"),
        Path("/usr/local/bin/methylGrapher"),
    ):
        if cand.is_file():
            return str(cand)
    which = shutil.which("methylGrapher")
    if which and Path(which).exists():
        return which
    return None


def companion_gfa_for_index(index_prefix: str) -> Optional[str]:
    explicit = os.environ.get("METHYLGRAPHER_GIRAFFE_GFA", "").strip()
    if explicit and Path(explicit).is_file():
        return explicit
    p = Path(index_prefix)
    name = p.name
    for suffix in (".wl.C2T", ".wl.G2A", ".C2T", ".G2A"):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            cand = p.parent / f"{base}.wl.gfa"
            if cand.is_file():
                return str(cand)
            cand2 = p.parent / f"{base}.gfa"
            if cand2.is_file():
                return str(cand2)
    for cand in (p.parent / f"{name}.gfa", p.with_suffix(".gfa")):
        if cand.is_file():
            return str(cand)
    parent = p.parent
    stem = name
    if ".wl." in stem:
        stem = stem.split(".wl.")[0] + ".wl"
        cand = parent / f"{stem}.gfa"
        if cand.is_file():
            return str(cand)
    return None


def gfa_usable_for_mojo(gfa_path: str) -> bool:
    path = Path(gfa_path)
    if not path.is_file():
        return False
    try:
        max_b = int(
            os.environ.get(
                "METHYLGRAPHER_MOJO_GIRAFFE_MAX_GFA_BYTES",
                str(_DEFAULT_MAX_GFA_BYTES),
            )
        )
    except ValueError:
        max_b = _DEFAULT_MAX_GFA_BYTES
    if max_b <= 0:
        return True
    return path.stat().st_size <= max_b


def parse_fq_from_giraffe_input(giraffe_input: str) -> tuple[Optional[str], Optional[str]]:
    parts = giraffe_input.split()
    fqs: list[str] = []
    i = 0
    while i < len(parts):
        if parts[i] == "-f" and i + 1 < len(parts):
            fqs.append(parts[i + 1])
            i += 2
            continue
        i += 1
    fq1 = fqs[0] if fqs else None
    fq2 = fqs[1] if len(fqs) > 1 else None
    return fq1, fq2


def build_mojo_giraffe_gfa_cmd(
    *,
    mojo_bin: str,
    gfa_path: str,
    fq1: str,
    fq2: Optional[str],
    device: Optional[str] = None,
    k: Optional[int] = None,
    out_gaf: Optional[str] = None,
) -> str:
    if not fq1:
        raise RuntimeError("MojoGiraffe requires fq1")
    dev = (device or os.environ.get("METHYLGRAPHER_GIRAFFE_DEVICE", "auto")).strip()
    k_arg = ""
    if k is not None:
        k_arg = f" -k {int(k)}"
    elif os.environ.get("METHYLGRAPHER_GIRAFFE_K", "").strip():
        k_arg = f" -k {os.environ['METHYLGRAPHER_GIRAFFE_K'].strip()}"
    fq2_arg = f" -fq2 {fq2}" if fq2 else ""
    # Prefer a real GAF path: Align's fd3/pipe + underscore-shard router drops
    # Illumina qnames from MojoGiraffe. File + stderr logs is the production path.
    out = (out_gaf or "").strip()
    env_prefix = 'METHYLGRAPHER_ENGINE=mojo '
    if out:
        return (
            "set -euo pipefail; "
            f'{env_prefix}"{mojo_bin}" MojoGiraffe -gfa "{gfa_path}" -fq1 "{fq1}"{fq2_arg} '
            f'-out_gaf "{out}" -device "{dev}"{k_arg} 1>&2'
        )
    return (
        "set -euo pipefail; "
        f'{env_prefix}"{mojo_bin}" MojoGiraffe -gfa "{gfa_path}" -fq1 "{fq1}"{fq2_arg} '
        f'-out_gaf /dev/fd/3 -device "{dev}"{k_arg} 3>&1 1>&2'
    )


# Back-compat alias
build_mojo_giraffe_gaf_cmd = build_mojo_giraffe_gfa_cmd


def build_mojo_giraffe_gbz_cmd(
    *,
    mojo_bin: str,
    gbz: str,
    dist: str,
    min_path: str,
    zipcodes: str,
    fq1: str,
    fq2: Optional[str],
    device: Optional[str] = None,
    k: Optional[int] = None,
    out_gaf: Optional[str] = None,
) -> str:
    if not fq1:
        raise RuntimeError("MojoGiraffe GBZ mode requires fq1")
    dev = (device or os.environ.get("METHYLGRAPHER_GIRAFFE_DEVICE", "auto")).strip()
    k_arg = ""
    if k is not None:
        k_arg = f" -k {int(k)}"
    elif os.environ.get("METHYLGRAPHER_GIRAFFE_K", "").strip():
        k_arg = f" -k {os.environ['METHYLGRAPHER_GIRAFFE_K'].strip()}"
    fq2_arg = f" -fq2 {fq2}" if fq2 else ""
    zip_arg = f' -zipcodes "{zipcodes}"' if zipcodes else ""
    out = (out_gaf or "").strip()
    # MojoGiraffe lives in the Mojo CLI (methylgrapher/src/main.mojo), not
    # the Python engine. Force METHYLGRAPHER_ENGINE=mojo for host + image bins.
    env_prefix = 'METHYLGRAPHER_ENGINE=mojo '
    if out:
        return (
            "set -euo pipefail; "
            f'{env_prefix}"{mojo_bin}" MojoGiraffe -gbz "{gbz}" -dist "{dist}" -min "{min_path}"'
            f"{zip_arg} -fq1 \"{fq1}\"{fq2_arg} "
            f'-out_gaf "{out}" -device "{dev}"{k_arg} 1>&2'
        )
    return (
        "set -euo pipefail; "
        f'{env_prefix}"{mojo_bin}" MojoGiraffe -gbz "{gbz}" -dist "{dist}" -min "{min_path}"'
        f"{zip_arg} -fq1 \"{fq1}\"{fq2_arg} "
        f'-out_gaf /dev/fd/3 -device "{dev}"{k_arg} 3>&1 1>&2'
    )


def resolve_map_command(
    *,
    align_engine: Optional[str],
    vg_path: str,
    thread: int,
    output_format: str,
    index_params: str,
    giraffe_input: str,
    index_prefix: Optional[str] = None,
    gfa_path: Optional[str] = None,
    out_gaf: Optional[str] = None,
) -> tuple[str, str]:
    """Return ``(engine_used, shell_command)`` for one dual-graph map invocation."""
    engine = normalize_align_engine(align_engine)
    if engine == "cpu_vg":
        return engine, build_vg_giraffe_gaf_cmd(
            vg_path=vg_path,
            thread=thread,
            output_format=output_format,
            index_params=index_params,
            giraffe_input=giraffe_input,
        )

    fq1, fq2 = parse_fq_from_giraffe_input(giraffe_input)
    gfa = gfa_path
    if not gfa and index_prefix:
        gfa = companion_gfa_for_index(index_prefix)
    if not gfa:
        gfa = os.environ.get("METHYLGRAPHER_GIRAFFE_GFA", "").strip() or None

    quartet = resolve_gbz_quartet(index_prefix) if index_prefix else None

    def _try_mojo_gbz(label: str) -> Optional[tuple[str, str]]:
        bin_path = resolve_mojo_giraffe_bin()
        if not bin_path or not quartet or not fq1:
            return None
        # READY defaults on (METHYLGRAPHER_MOJO_GIRAFFE_READY=1). Opt out with
        # 0/false/off to force vg while Buffy wall / DS20M gates are pending.
        if not mojo_giraffe_ready():
            return None
        gbz_path = quartet["gbz"]
        # Production GBZ (~GB) must have a prebuilt segment pack; otherwise Align
        # would spend hours in `vg convert` / OOM building cache mid-map.
        max_direct = int(
            os.environ.get(
                "METHYLGRAPHER_MOJO_GBZ_DIRECT_MAX_BYTES",
                str(64 * 1024 * 1024),
            )
        )
        try:
            gbz_bytes = Path(gbz_path).stat().st_size
        except OSError:
            return None
        if gbz_bytes > max_direct and not segment_cache_ready(gbz_path):
            return None
        note = (
            f"# {label} mojo_giraffe_gbz host={platform.machine()} "
            f"device={os.environ.get('METHYLGRAPHER_GIRAFFE_DEVICE', 'auto')} "
            f"ready=1 gbz={gbz_path}\n"
        )
        cmd = build_mojo_giraffe_gbz_cmd(
            mojo_bin=bin_path,
            gbz=gbz_path,
            dist=quartet["dist"],
            min_path=quartet["min"],
            zipcodes=quartet.get("zipcodes") or "",
            fq1=fq1,
            fq2=fq2,
            out_gaf=out_gaf,
        )
        return f"{label}+mojo_gbz", note + cmd

    def _try_mojo_gfa(label: str) -> Optional[tuple[str, str]]:
        bin_path = resolve_mojo_giraffe_bin()
        if not bin_path or not gfa or not fq1:
            return None
        if not gfa_usable_for_mojo(gfa):
            return None
        note = (
            f"# {label} mojo_giraffe_gfa host={platform.machine()} "
            f"device={os.environ.get('METHYLGRAPHER_GIRAFFE_DEVICE', 'auto')} "
            f"gfa={gfa}\n"
        )
        cmd = build_mojo_giraffe_gfa_cmd(
            mojo_bin=bin_path,
            gfa_path=gfa,
            fq1=fq1,
            fq2=fq2,
            out_gaf=out_gaf,
        )
        return f"{label}+mojo_gfa", note + cmd

    def _try_mojo(label: str) -> Optional[tuple[str, str]]:
        # Production: GBZ quartet first (no GFA size cap). Fixture GFA second.
        return _try_mojo_gbz(label) or _try_mojo_gfa(label)

    if engine == "mojo_giraffe":
        got = _try_mojo("mojo_giraffe")
        if got:
            return got
        raise RuntimeError(
            "align_engine=mojo_giraffe requires MojoGiraffe binary + GBZ quartet "
            "(index_prefix.giraffe.gbz/.dist/.min/.zipcodes) or usable GFA. "
            "See docs/GIRAFFE_SPEC.md."
        )

    fallback = gpu_giraffe_fallback()
    if fallback == "mojo":
        got = _try_mojo("gpu_giraffe")
        if got:
            return got
        note = (
            f"# gpu_giraffe mojo unavailable; auto vg "
            f"host={platform.machine()} threads={thread} "
            f"gbz={'yes' if quartet else 'no'} gfa={gfa or 'none'}\n"
        )
        cmd = build_vg_giraffe_gaf_cmd(
            vg_path=vg_path,
            thread=thread,
            output_format=output_format,
            index_params=index_params,
            giraffe_input=giraffe_input,
        )
        return "gpu_giraffe+vg_autoscale", note + cmd

    if fallback == "error":
        raise RuntimeError(
            "align_engine=gpu_giraffe but Mojo Giraffe GAF path is unavailable "
            "and METHYLGRAPHER_GPU_GIRAFFE_FALLBACK=error. Set fallback=mojo "
            "or fallback=vg. See docs/GIRAFFE_SPEC.md."
        )

    note = (
        f"# gpu_giraffe fallback=vg host={platform.machine()} "
        f"threads={thread}\n"
    )
    cmd = build_vg_giraffe_gaf_cmd(
        vg_path=vg_path,
        thread=thread,
        output_format=output_format,
        index_params=index_params,
        giraffe_input=giraffe_input,
    )
    return "gpu_giraffe+vg_fallback", note + cmd
