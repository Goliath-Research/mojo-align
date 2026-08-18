# Mojo-native vg MinimizerIndex v11 (Q1Q1) HT probe over mmap.
#
# Open is a thin bridge (engine.min_mmap_bridge) so the mapping stays alive;
# all header parse + quadratic probe + hit decode runs in Mojo via UnsafePointer.

from std.collections import List
from std.memory import UnsafePointer
from std.python import Python, PythonObject

from giraffe_minimizer import MinimizerOcc, wang_hash_64
from mojo_align_env import ensure_python_path


comptime TAG_Q1Q1 = 0x31513151
comptime NO_KEY = UInt64(0x7FFFFFFFFFFFFFFF)
comptime IS_POINTER = UInt64(1) << 63
comptime OFFSET_BITS = 10
comptime REV_MASK = UInt64(1) << OFFSET_BITS
comptime OFF_MASK = REV_MASK - 1


struct MojoMinIndex(Movable):
    """Mapped ``.shortread.withzip.min`` — Mojo HT probe (unique cells)."""

    var _keep: PythonObject
    var addr: Int
    var size: Int
    var k: Int
    var w: Int
    var cell_size: Int
    var cell_count: Int
    var ht_data_offset: Int
    var payload_size: Int

    def __init__(out self, path: String) raises:
        ensure_python_path()
        var bridge = Python.import_module("engine.min_mmap_bridge")
        var opened = bridge.open_min_mmap(path)
        self.addr = Int(py=opened[0])
        self.size = Int(py=opened[1])
        self._keep = Python.tuple(opened[2], opened[3])
        if self.addr == 0 or self.size < 120:
            raise Error("min mmap open failed: " + path)

        var base = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=self.addr)
        var tag = (
            Int(base[0])
            | (Int(base[1]) << 8)
            | (Int(base[2]) << 16)
            | (Int(base[3]) << 24)
        )
        if tag != TAG_Q1Q1:
            raise Error("not a MinimizerIndex Q1Q1 at " + path)

        var u64p = base.bitcast[UInt64]()
        # Layout: u32 tag, u32 version, then 8×u64 at offset 8 → u64 index 1..
        self.k = Int(u64p[1])
        self.w = Int(u64p[2])
        var flags = u64p[8]
        var key_bits = Int(flags & 0xFF)
        self.payload_size = Int((flags >> 12) & 0xF)
        if key_bits != 64:
            raise Error("only 64-bit minimizer keys supported")
        self.cell_size = 1 + 1 + self.payload_size
        # ht_words at byte offset 112 → u64 index 14
        var ht_words = Int(u64p[14])
        if ht_words % self.cell_size != 0:
            raise Error("ht_words not divisible by cell_size")
        self.cell_count = ht_words // self.cell_size
        if self.cell_count == 0 or (self.cell_count & (self.cell_count - 1)) != 0:
            raise Error("cell_count must be power-of-two")
        self.ht_data_offset = 120

    def close(mut self) raises:
        if self.addr == 0:
            return
        ensure_python_path()
        var bridge = Python.import_module("engine.min_mmap_bridge")
        bridge.close_min_mmap(self._keep[0], self._keep[1])
        self.addr = 0
        self.size = 0

    def _base(self) raises -> UnsafePointer[UInt8, MutAnyOrigin]:
        return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=self.addr)

    def _load_u64(self, byte_off: Int) raises -> UInt64:
        var p = UnsafePointer[UInt64, MutAnyOrigin](
            unsafe_from_address=self.addr + byte_off
        )
        return p[]

    def find_offset(self, key_in: UInt64) raises -> Int:
        """Return HT array_off (words) or -1."""
        var key = key_in & NO_KEY
        var h = wang_hash_64(key)
        var cell_count = self.cell_count
        var cell_offset = Int(h) & (cell_count - 1)
        var attempt = 0
        while attempt < cell_count:
            var array_off = cell_offset * self.cell_size
            var cell_key = self._load_u64(self.ht_data_offset + array_off * 8)
            var bare = cell_key & NO_KEY
            if bare == NO_KEY or bare == key:
                if bare == NO_KEY:
                    return -1
                return array_off
            cell_offset = (cell_offset + attempt + 1) & (cell_count - 1)
            attempt += 1
        return -1

    def hits_at(self, array_off: Int, hit_cap: Int) raises -> List[String]:
        """Decode unique-cell hits as ``node:orient:offset`` (skip pointer cells)."""
        var out = List[String]()
        var key = self._load_u64(self.ht_data_offset + array_off * 8)
        if (key & IS_POINTER) != 0:
            # Multi-hit pointer lists require a side index; unique keys dominate.
            return out^
        var pos = self._load_u64(self.ht_data_offset + (array_off + 1) * 8)
        var node_id = Int(pos >> UInt64(OFFSET_BITS + 1))
        var is_rev = (pos & REV_MASK) != 0
        var offset = Int(pos & OFF_MASK)
        if node_id == 0:
            return out^
        var orient = String("0")
        if is_rev:
            orient = String("1")
        out.append(String(node_id) + ":" + orient + ":" + String(offset))
        _ = hit_cap
        return out^

    def locate_keys(
        self, keys: List[UInt64], hit_cap: Int = 24
    ) raises -> List[String]:
        var out = List[String]()
        var seen = List[String]()
        for key in keys:
            var off = self.find_offset(key)
            if off < 0:
                continue
            var hits = self.hits_at(off, hit_cap)
            for h in hits:
                var dup = False
                for s in seen:
                    if s == h:
                        dup = True
                        break
                if dup:
                    continue
                seen.append(h.copy())
                out.append(h.copy())
                if len(out) >= hit_cap * 4:
                    return out^
        return out^

    def locate_occs_batch(
        self, occs_batch: List[List[MinimizerOcc]], hit_cap: Int = 24
    ) raises -> List[List[String]]:
        var out = List[List[String]]()
        for occs in occs_batch:
            var keys = List[UInt64]()
            for occ in occs:
                keys.append(occ.key)
            out.append(self.locate_keys(keys, hit_cap))
        return out^
