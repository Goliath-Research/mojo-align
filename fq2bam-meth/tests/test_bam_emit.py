"""BAM emit: BGZF framing + samtools-readable records."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

from bam_emit import BamArena, BamWriter, _pack_seq, _reg2bin


def test_pack_seq_and_bin():
    assert _pack_seq(b"ACGT") == bytes([0x12, 0x48])  # A=1 C=2 G=4 T=8
    assert _reg2bin(0, 1) == ((1 << 15) - 1) // 7


@pytest.mark.skipif(shutil.which("samtools") is None, reason="samtools required")
def test_bam_writer_roundtrip(tmp_path: Path):
    bam = tmp_path / "t.bam"
    w = BamWriter(
        str(bam), ["chr1"], [1000], rg_id="mojo1", level=1, sort_order="coordinate"
    )
    w.write_batch(
        qnames=["r1", "r1"],
        flags=[99, 147],
        tids=[0, 0],
        pos0s=[10, 50],
        mapqs=[60, 60],
        sls=[0, 0],
        srs=[0, 0],
        seqs=["ACGTACGTAC", "TGCATGCATG"],
        quals=["IIIIIIIIII", "IIIIIIIIII"],
        next_tids=[0, 0],
        next_pos0s=[50, 10],
        tlens=[50, -50],
        nms=[0, 1],
    )
    w.close()
    view = subprocess.run(
        ["samtools", "view", "-h", str(bam)],
        check=True,
        capture_output=True,
        text=True,
    )
    text = view.stdout
    assert "@HD\tVN:1.6\tSO:coordinate" in text
    assert "@SQ\tSN:chr1\tLN:1000" in text
    assert "@RG\tID:mojo1" in text
    lines = [ln for ln in text.splitlines() if not ln.startswith("@")]
    assert len(lines) == 2
    p = lines[0].split("\t")
    assert p[0] == "r1"
    assert p[1] == "99"
    assert p[2] == "chr1"
    assert p[3] == "11"  # 1-based
    assert p[5] == "10M"
    assert p[9] == "ACGTACGTAC"
    assert "RG:Z:mojo1" in lines[0]
    assert "NM:i:0" in lines[0]
    fs = subprocess.run(
        ["samtools", "flagstat", str(bam)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "2 + 0 mapped" in fs.stdout


def test_bam_arena_chunks():
    payload = bytearray(b"defgh")
    buf = (ctypes.c_char * 5).from_buffer(payload)
    arena = BamArena()
    idx = arena.append_raw(ctypes.addressof(buf), 5)
    assert idx == 0
    assert arena.n_chunks() == 1
    assert arena.chunk_len(0) == 5
    assert arena.nbytes() == 5
    assert ctypes.string_at(arena.chunk_addr(0), 5) == b"defgh"


def test_bam_arena_spill_mmap(tmp_path: Path):
    payload = bytearray(b"spillok")
    buf = (ctypes.c_char * 7).from_buffer(payload)
    arena = BamArena(spill_dir=tmp_path)
    idx = arena.append_raw(ctypes.addressof(buf), 7)
    assert idx == 0
    assert (tmp_path / "chunk_000000.bin").is_file()
    assert arena.chunk_len(0) == 7
    assert ctypes.string_at(arena.chunk_addr(0), 7) == b"spillok"
    empty = arena.append_raw(0, 0)
    assert empty == 1
    assert arena.chunk_len(1) == 0
    assert arena.n_chunks() == 2


def test_order_run_perms():
    from bam_emit import order_run_perms

    n = 5
    coord, ca = _u64([50, 10, 40, 20, 30])
    dhi, hia = _u64([9, 1, 1, 2, 8])
    dlo, loa = _u64([0, 5, 3, 0, 0])
    perm, pma = _u32([0, 0, 0, 0, 0])
    dup, dpa = _u32([0, 0, 0, 0, 0])
    order_run_perms(n, ca, hia, loa, pma, dpa)
    assert list(perm) == [1, 3, 4, 2, 0]
    assert list(dup) == [2, 1, 3, 4, 0]
    _ = (coord, dhi, dlo)


def test_bam_arena_clear(tmp_path: Path):
    payload = bytearray(b"xyz")
    buf = (ctypes.c_char * 3).from_buffer(payload)
    arena = BamArena(spill_dir=tmp_path)
    arena.append_raw(ctypes.addressof(buf), 3)
    assert arena.n_chunks() == 1
    arena.clear()
    assert arena.n_chunks() == 0
    assert arena.nbytes() == 0
    assert not (tmp_path / "chunk_000000.bin").exists()


def _u64(vals: list[int]):
    a = (ctypes.c_uint64 * len(vals))(*vals)
    return a, ctypes.addressof(a)


def _u32(vals: list[int]):
    a = (ctypes.c_uint32 * len(vals))(*vals)
    return a, ctypes.addressof(a)


class _ByteSink:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write_bytes(self, data: bytes | bytearray | memoryview) -> None:
        self.chunks.append(bytes(data))


def _fake_bam_rec(tid: int, pos: int) -> bytes:
    """Minimal uncompressed BAM: block_size + 32-byte core (tid/pos at +4/+8)."""
    import struct

    body = struct.pack("<iiIIiiii", tid, pos, 0, 0, 0, -1, -1, 0)
    return struct.pack("<i", len(body)) + body


def test_write_sorted_from_arena_uses_packed_tid_pos(tmp_path: Path):
    from bam_emit import BamRunStore

    recs = [
        _fake_bam_rec(2, 50),
        _fake_bam_rec(0, 10),
        _fake_bam_rec(-1, 0),
        _fake_bam_rec(0, 5),
    ]
    blob = b"".join(recs)
    buf = ctypes.create_string_buffer(blob)
    arena = BamArena(spill_dir=tmp_path / "arena")
    arena.append_raw(ctypes.addressof(buf), len(blob))
    n = len(recs)
    perm, pma = _u32([0] * n)
    rec_dummy, ra = _u64([0] * n)
    ln_dummy, la = _u32([0] * n)
    store = BamRunStore(spill_dir=tmp_path)
    store.write_sorted_from_arena(n, arena, ra, la, pma)
    assert list(perm) == [3, 1, 0, 2]
    bam_path = tmp_path / "sorted_runs" / "run_000000.bam.bin"
    raw = bam_path.read_bytes()
    assert raw == recs[3] + recs[1] + recs[0] + recs[2]
    _ = (rec_dummy, ln_dummy)


def test_bam_run_store_merge_and_markdup(tmp_path: Path):
    from bam_emit import BamRunStore, _or_dup_flag_bytes

    store = BamRunStore(spill_dir=tmp_path)
    recs0 = [_fake_bam_rec(0, 100), _fake_bam_rec(0, 300)]
    recs1 = [_fake_bam_rec(0, 200), _fake_bam_rec(0, 400)]

    def add(recs, coords, dhis, dlos, scores, pairs, perm, dupperm):
        n = len(recs)
        w = store.open_raw_writer()
        ordered = b"".join(recs[p] for p in perm)
        buf = ctypes.create_string_buffer(ordered)
        w.write_raw(ctypes.addressof(buf), len(ordered))
        w.close()
        c, ca = _u64(coords)
        hi, hia = _u64(dhis)
        lo, loa = _u64(dlos)
        sc, sca = _u32(scores)
        pr, pra = _u32(pairs)
        ln, lna = _u32([len(r) for r in recs])
        pm, pma = _u32(perm)
        dp, dpa = _u32(dupperm)
        store.write_keys(n, ca, hia, loa, sca, pra, lna, pma, dpa)
        return c, hi, lo, sc, pr, ln, pm, dp

    keep0 = add(
        recs0,
        coords=[100, 300],
        dhis=[1, 9],
        dlos=[1, 9],
        scores=[100, 1],
        pairs=[0, 2],
        perm=[0, 1],
        dupperm=[0, 1],
    )
    keep1 = add(
        recs1,
        coords=[200, 400],
        dhis=[1, 8],
        dlos=[1, 8],
        scores=[50, 1],
        pairs=[1, 3],
        perm=[0, 1],
        dupperm=[0, 1],
    )
    _ = (keep0, keep1)
    sink = _ByteSink()
    marked = store.merge_into(sink, do_markdup=True)
    assert marked == 1
    dup = bytearray(recs1[0])
    _or_dup_flag_bytes(dup)
    out = b"".join(sink.chunks)
    assert out == recs0[0] + bytes(dup) + recs0[1] + recs1[1]
