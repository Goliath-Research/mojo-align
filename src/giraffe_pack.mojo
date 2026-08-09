# Dense segment pack reader (dense-v1) — Mojo mmap hot path.
#
# Production packs are contiguous node ids 1..N. offsets.bin + sequences.bin
# are libc-mmap'd once; ``get`` reads u64 (offset,length) and copies ASCII
# into a String without calling Python SegmentPack.get.

from std.collections import List
from std.memory import UnsafePointer
from std.python import Python, PythonObject


def _bootstrap_sys_path() raises:
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, String(os_mod.getcwd()))
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")


def _ascii_from_addr(addr: Int, length: Int) raises -> String:
    if length <= 0:
        return String("")
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)
    var out = String("")
    var i = 0
    while i < length:
        out += chr(Int(p[i]))
        i += 1
    return out^


struct DensePack(Copyable, Movable):
    """Host-side view of ``*.mojo_segments`` (Mojo mmap for contiguous dense-v1)."""

    var root: String
    var n: Int
    var contiguous: Bool
    var off_addr: Int
    var off_size: Int
    var seq_addr: Int
    var seq_size: Int
    var _keep_off: PythonObject
    var _keep_seq: PythonObject
    var _pack: PythonObject
    var _use_py: Bool

    def __init__(out self, root: String) raises:
        self.root = root
        self.n = 0
        self.contiguous = False
        self.off_addr = 0
        self.off_size = 0
        self.seq_addr = 0
        self.seq_size = 0
        self._keep_off = Python.none()
        self._keep_seq = Python.none()
        self._pack = Python.none()
        self._use_py = True

        _bootstrap_sys_path()
        var os_mod = Python.import_module("os")
        var off_path = root + "/offsets.bin"
        var seq_path = root + "/sequences.bin"
        var ids_path = root + "/ids.txt"
        if (
            not Bool(os_mod.path.isfile(off_path))
            or not Bool(os_mod.path.isfile(seq_path))
        ):
            var sp = Python.import_module("engine.segment_pack")
            self._pack = sp.SegmentPack(root)
            self._use_py = True
            self.n = Int(py=self._pack.__len__())
            return

        var bridge = Python.import_module("engine.min_mmap_bridge")
        var opened_off = bridge.open_min_mmap(off_path)
        var opened_seq = bridge.open_min_mmap(seq_path)
        self.off_addr = Int(py=opened_off[0])
        self.off_size = Int(py=opened_off[1])
        self._keep_off = Python.tuple(opened_off[2], opened_off[3])
        self.seq_addr = Int(py=opened_seq[0])
        self.seq_size = Int(py=opened_seq[1])
        self._keep_seq = Python.tuple(opened_seq[2], opened_seq[3])
        if self.off_addr == 0 or self.seq_addr == 0 or self.off_size < 16:
            raise Error("dense pack mmap failed: " + root)

        self.n = self.off_size // 16
        # Contiguous 1..N probe once at open (Python helper; not per-get).
        _ = ids_path
        self.contiguous = Bool(bridge.probe_dense_contiguous(root, self.n))

        if self.contiguous:
            self._use_py = False
            print(
                "dense_pack mojo_mmap contiguous n=",
                self.n,
                " root=",
                root,
                flush=True,
            )
            return

        # Non-contiguous / jsonl: keep Python SegmentPack for id→index map.
        var sp2 = Python.import_module("engine.segment_pack")
        self._pack = sp2.SegmentPack(root)
        self._use_py = True
        self.n = Int(py=self._pack.__len__())
        print(
            "dense_pack python_fallback n=",
            self.n,
            " root=",
            root,
            flush=True,
        )

    def close(mut self) raises:
        if self.off_addr == 0 and self.seq_addr == 0:
            return
        _bootstrap_sys_path()
        var bridge = Python.import_module("engine.min_mmap_bridge")
        if self.off_addr != 0:
            bridge.close_min_mmap(self._keep_off[0], self._keep_off[1])
            self.off_addr = 0
        if self.seq_addr != 0:
            bridge.close_min_mmap(self._keep_seq[0], self._keep_seq[1])
            self.seq_addr = 0

    def _load_u64(self, base_addr: Int, byte_off: Int) raises -> UInt64:
        var p = UnsafePointer[UInt64, MutAnyOrigin](
            unsafe_from_address=base_addr + byte_off
        )
        return p[]

    def get_by_index(self, idx: Int) raises -> String:
        if idx < 0 or idx >= self.n:
            return String("")
        if self._use_py:
            var sid = String(self._pack.ids()[idx])
            var seg = self._pack.get(sid)
            if seg is None:
                return String("")
            return String(seg)
        var base = idx * 16
        var offset = Int(self._load_u64(self.off_addr, base))
        var length = Int(self._load_u64(self.off_addr, base + 8))
        if offset < 0 or length < 0 or offset + length > self.seq_size:
            return String("")
        return _ascii_from_addr(self.seq_addr + offset, length)

    def get(self, node_id: Int) raises -> String:
        if self._use_py:
            var seg = self._pack.get(String(node_id))
            if seg is None:
                return String("")
            return String(seg)
        if not self.contiguous:
            return String("")
        if node_id < 1 or node_id > self.n:
            return String("")
        return self.get_by_index(node_id - 1)

    def get_sid(self, sid: String) raises -> String:
        if self._use_py:
            var seg = self._pack.get(sid)
            if seg is None:
                return String("")
            return String(seg)
        var nid = Int(sid)
        return self.get(nid)

    def size(self) raises -> Int:
        return self.n

    def segment_ids(self) raises -> List[String]:
        """Toy/fixture only — do not call on production 100M+ packs."""
        var out = List[String]()
        if self._use_py:
            var ids = self._pack.ids()
            var n = Int(py=ids.__len__())
            var i = 0
            while i < n:
                out.append(String(ids[i]))
                i += 1
            return out^
        var j = 1
        while j <= self.n:
            out.append(String(j))
            j += 1
        return out^


def reverse_complement_dna(seq: String) raises -> String:
    var out = String("")
    var n = seq.byte_length()
    var i = n - 1
    while i >= 0:
        var b = String(seq[byte = i : i + 1]).upper()
        if b == "A":
            out += "T"
        elif b == "T":
            out += "A"
        elif b == "C":
            out += "G"
        elif b == "G":
            out += "C"
        else:
            out += "N"
        i -= 1
    return out^
