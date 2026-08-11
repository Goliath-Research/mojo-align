# Minimizer + zipcode locate for GBZ-native Mojo Giraffe.
#
# Production science locate:
#   1) Mojo Giraffe (k,w) minimizers — DeviceContext pack/hash on NVIDIA/AMD
#      (host buffers → window reduce; never CuPy / huge List copies)
#   2) Mojo-native Q1Q1 HT probe over mmap (``giraffe_min_index.MojoMinIndex``)

from std.collections import Dict, List
from std.python import Python

from giraffe_min_index import MojoMinIndex
from giraffe_minimizer import minimizers_batch


def probe_min_index(min_path: String) raises -> String:
    var idx = MojoMinIndex(min_path)
    var s = (
        "k="
        + String(idx.k)
        + " w="
        + String(idx.w)
        + " cells="
        + String(idx.cell_count)
    )
    idx.close()
    return s


def locate_read_hits(min_path: String, seq: String, hit_cap: Int = 24) raises -> List[String]:
    var seqs = List[String]()
    seqs.append(seq)
    var batch = locate_batch_hits_native("cpu", min_path, seqs, hit_cap)
    if len(batch) == 0:
        return List[String]()
    return batch[0].copy()


def locate_batch_hits(
    min_path: String, seqs: List[String], hit_cap: Int = 24
) raises -> List[List[String]]:
    return locate_batch_hits_native("cpu", min_path, seqs, hit_cap)


def locate_occs_label(backend: String) raises -> String:
    var label = backend + "+mojo_min_mmap+mojo_pack+mojo_cluster+mojo_stream"
    var os_mod = Python.import_module("os")
    var prev = String(os_mod.environ.get("METHYLGRAPHER_LAST_SEED_BACKEND", ""))
    os_mod.environ["METHYLGRAPHER_LAST_SEED_BACKEND"] = label
    if prev != label:
        print("mojo_stream seed_backend=", label, flush=True)
    return label


def locate_batch_hits_native(
    device: String,
    min_path: String,
    seqs: List[String],
    hit_cap: Int = 24,
) raises -> List[List[String]]:
    """Mojo minimizers + Mojo mmap HT probe. Opens the index once per call."""
    if min_path.byte_length() == 0 or len(seqs) == 0:
        var empty = List[List[String]]()
        for _s in seqs:
            empty.append(List[String]())
        return empty^

    var idx = MojoMinIndex(min_path)
    var result = minimizers_batch(device, seqs, idx.k, idx.w)
    var backend = result.backend.copy()
    var out = idx.locate_occs_batch(result.occs, hit_cap)
    idx.close()
    _ = locate_occs_label(backend)
    return out^


def locate_batch_hits_with_index(
    mut idx: MojoMinIndex,
    device: String,
    seqs: List[String],
    hit_cap: Int = 24,
) raises -> List[List[String]]:
    """Same as native locate but reuses an already-open ``MojoMinIndex``.

    Prefer ``map_fastq_stream_to_gaf`` GPU session (one DeviceContext for the
    whole FASTQ). This entry still constructs a DeviceContext per call — fine
    for smokes / tiny batches, not for production multi-GB FASTQs.
    """
    if len(seqs) == 0:
        return List[List[String]]()
    var result = minimizers_batch(device, seqs, idx.k, idx.w)
    var backend = result.backend.copy()
    var out = idx.locate_occs_batch(result.occs, hit_cap)
    _ = locate_occs_label(backend)
    return out^


def build_min_index_from_segments(
    segments: Dict[String, String], k: Int
) raises -> List[String]:
    raise Error(
        "build_min_index_from_segments removed; use .shortread.withzip.min via locate_read_hits"
    )
