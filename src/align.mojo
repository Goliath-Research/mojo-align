# src/align.mojo
# Mojo Align orchestration: parse CLI, select map backend
# (cpu_vg | gpu_giraffe | mojo_giraffe), then run engine.alignments.alignment_main.
# gpu_giraffe prefers MojoGiraffe on a ready GBZ quartet (or usable GFA);
# auto-vg / FALLBACK=vg for emergency rollback. Parabricks remains Phase 0
# NO-GO for science GAF.

from std.collections import List
from std.python import Python

from utility import bool_from_str, get_kv_value


def run_align_native(args: List[String]) raises -> Int:
    """Native Mojo control plane for Align; map kernel via align_backends."""
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")

    var repo_root = os_mod.getcwd()
    sys_mod.path.insert(0, repo_root)

    var fq1 = get_kv_value(args, "fq1", "")
    var fq2_raw = get_kv_value(args, "fq2", "")
    var work_dir = get_kv_value(args, "work_dir", "./")
    var index_prefix = get_kv_value(args, "index_prefix", "")
    var thread = Int(get_kv_value(args, "t", "1"))
    var compress = bool_from_str(get_kv_value(args, "compress", "N"))
    var directional = bool_from_str(get_kv_value(args, "directional", "Y"))
    var vg_path = get_kv_value(args, "vg_path", "vg")
    var align_engine = get_kv_value(args, "align_engine", "")

    if fq1.byte_length() == 0 or index_prefix.byte_length() == 0:
        raise Error("Align requires -fq1 and -index_prefix")

    # Prefer explicit CLI; else env (worker/site); else cpu_vg inside normalize.
    if align_engine.byte_length() == 0:
        align_engine = String(os_mod.environ.get("METHYLGRAPHER_ALIGN_ENGINE", ""))

    var backends = Python.import_module("engine.align_backends")
    var engine_arg = Python.none()
    if align_engine.byte_length() > 0:
        engine_arg = align_engine
    var engine_norm = backends.normalize_align_engine(engine_arg)
    print("Mojo Align orchestration; align_engine=" + String(engine_norm))

    var alignments = Python.import_module("engine.alignments")
    var fq2_obj = Python.none()
    if fq2_raw.byte_length() > 0:
        fq2_obj = fq2_raw

    alignments.alignment_main(
        fq1,
        fq2_obj,
        work_dir,
        index_prefix,
        compress,
        thread,
        directional,
        vg_path,
        engine_norm,
    )

    sys_mod.stdout.flush()
    sys_mod.stderr.flush()
    return 0
