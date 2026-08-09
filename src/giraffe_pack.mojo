# Dense segment pack reader (dense-v1) for Mojo gapless extend.

from std.collections import Dict, List
from std.python import Python, PythonObject


struct DensePack(Copyable, Movable):
    """Host-side view of ``*.mojo_segments`` (ids + sequences via Python mmap)."""

    var root: String
    var _pack: PythonObject

    def __init__(out self, root: String) raises:
        self.root = root
        var os_mod = Python.import_module("os")
        var sys_mod = Python.import_module("sys")
        sys_mod.path.insert(0, String(os_mod.getcwd()))
        sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
        sys_mod.path.insert(0, "/home/ubuntu/methylGrapher-mojo")
        var sp = Python.import_module("engine.segment_pack")
        self._pack = sp.SegmentPack(root)

    def get(self, node_id: Int) raises -> String:
        var seg = self._pack.get(String(node_id))
        if seg is None:
            return String("")
        return String(seg)

    def get_sid(self, sid: String) raises -> String:
        var seg = self._pack.get(sid)
        if seg is None:
            return String("")
        return String(seg)

    def size(self) raises -> Int:
        return Int(py=self._pack.__len__())

    def segment_ids(self) raises -> List[String]:
        """Toy/fixture only — do not call on production 100M+ packs."""
        var out = List[String]()
        var ids = self._pack.ids()
        var n = Int(py=ids.__len__())
        var i = 0
        while i < n:
            out.append(String(ids[i]))
            i += 1
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
