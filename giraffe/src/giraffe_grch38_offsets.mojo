# Dense GRCh38 segment→(chrom,start,len) table for MojoGiraffe QC SAM emit.
#
# Layout (grch38-dense-v1) under ``root/``:
#   meta.json, chroms.tsv, records.bin
# records.bin: (max_id+1) × 16 bytes LE — u32 chrom_idx, u32 length, u64 start.
# chrom_idx == 0xFFFFFFFF means unset.

from std.collections import List
from std.memory import UnsafePointer
from std.python import Python, PythonObject

comptime UNSET_CHROM: UInt32 = 0xFFFFFFFF
comptime RECORD_BYTES: Int = 16


struct Grch38SegInfo(Copyable, Movable):
    var chrom: String
    var start0: Int
    var length: Int
    var ok: Bool

    def __init__(out self, chrom: String, start0: Int, length: Int, ok: Bool):
        self.chrom = chrom
        self.start0 = start0
        self.length = length
        self.ok = ok


struct Grch38Offsets(Copyable, Movable):
    var root: String
    var max_id: Int
    var addr: Int
    var size: Int
    var _keep: PythonObject
    var chroms: List[String]
    var chrom_lens: List[Int]
    var loaded: Bool

    def __init__(out self):
        self.root = ""
        self.max_id = 0
        self.addr = 0
        self.size = 0
        self._keep = Python.none()
        self.chroms = List[String]()
        self.chrom_lens = List[Int]()
        self.loaded = False

    def open(mut self, root: String) raises:
        var os_mod = Python.import_module("os")
        var json_mod = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var bridge = Python.import_module("engine.min_mmap_bridge")
        self.root = root
        var meta_path = root + "/meta.json"
        var rec_path = root + "/records.bin"
        var chrom_path = root + "/chroms.tsv"
        if not Bool(os_mod.path.isfile(meta_path)) or not Bool(
            os_mod.path.isfile(rec_path)
        ):
            raise Error("grch38 offsets missing meta/records under " + root)
        var meta_fh = builtins.open(meta_path, "r")
        var meta = json_mod.load(meta_fh)
        meta_fh.close()
        var fmt = String(meta["format"])
        if fmt != "grch38-dense-v1":
            raise Error("unsupported grch38 offsets format: " + fmt)
        self.max_id = Int(py=meta["max_id"])
        var opened = bridge.open_min_mmap(rec_path)
        self.addr = Int(py=opened[0])
        self.size = Int(py=opened[1])
        self._keep = Python.tuple(opened[2], opened[3])
        if self.addr == 0 or self.size < RECORD_BYTES:
            raise Error("grch38 offsets mmap failed: " + rec_path)
        var expect = (self.max_id + 1) * RECORD_BYTES
        if self.size < expect:
            raise Error(
                "grch38 offsets records.bin too small: "
                + String(self.size)
                + " < "
                + String(expect)
            )
        self.chroms = List[String]()
        self.chrom_lens = List[Int]()
        if Bool(os_mod.path.isfile(chrom_path)):
            var cfh = builtins.open(chrom_path, "r")
            while True:
                var line = String(cfh.readline())
                if line.byte_length() == 0:
                    break
                # strip newline
                while line.byte_length() > 0 and (
                    line.endswith("\n") or line.endswith("\r")
                ):
                    var trimmed = String(line[byte = 0 : line.byte_length() - 1])
                    line = trimmed
                if line.byte_length() == 0:
                    continue
                var tab = line.find("\t")
                if tab < 0:
                    self.chroms.append(line)
                    self.chrom_lens.append(0)
                else:
                    self.chroms.append(String(line[byte = 0:tab]))
                    var ln_s = String(line[byte = tab + 1 : line.byte_length()])
                    var ln = 0
                    try:
                        ln = Int(ln_s)
                    except:
                        ln = 0
                    self.chrom_lens.append(ln)
            cfh.close()
        self.loaded = True
        print(
            "grch38_offsets mojo_mmap max_id=",
            self.max_id,
            " chroms=",
            len(self.chroms),
            " root=",
            root,
            flush=True,
        )

    def close(mut self) raises:
        if self.addr == 0:
            return
        var bridge = Python.import_module("engine.min_mmap_bridge")
        bridge.close_min_mmap(self._keep[0], self._keep[1])
        self.addr = 0
        self.loaded = False

    def _load_u32(self, byte_off: Int) raises -> UInt32:
        var p = UnsafePointer[UInt32, MutAnyOrigin](
            unsafe_from_address=self.addr + byte_off
        )
        return p[]

    def _load_u64(self, byte_off: Int) raises -> UInt64:
        var p = UnsafePointer[UInt64, MutAnyOrigin](
            unsafe_from_address=self.addr + byte_off
        )
        return p[]

    def lookup(self, seg_id: Int) raises -> Grch38SegInfo:
        if not self.loaded or seg_id < 0 or seg_id > self.max_id:
            return Grch38SegInfo("", 0, 0, False)
        var base = seg_id * RECORD_BYTES
        var cidx = self._load_u32(base)
        if cidx == UNSET_CHROM:
            return Grch38SegInfo("", 0, 0, False)
        var ci = Int(cidx)
        if ci < 0 or ci >= len(self.chroms):
            return Grch38SegInfo("", 0, 0, False)
        var length = Int(self._load_u32(base + 4))
        var start0 = Int(self._load_u64(base + 8))
        return Grch38SegInfo(self.chroms[ci], start0, length, True)
