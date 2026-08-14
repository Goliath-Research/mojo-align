# Mojo mmap lookup for GBZ-node → GFA-segment named-coordinates.
# Layout matches giraffe/python/named_coords.py: nodes.bin is dense
# little-endian (uint32 seg_id, uint32 offset_in_segment) per node id,
# entry 0 unused.

from std.ffi import external_call
from std.memory import UnsafePointer
from std.python import Python, PythonObject


struct NamedCoordsMojo(Copyable, Movable):
    var addr: Int
    var nbytes: Int
    var fd: Int
    var n_nodes: Int
    var alive: Bool

    def __init__(out self):
        self.addr = 0
        self.nbytes = 0
        self.fd = -1
        self.n_nodes = 0
        self.alive = False


def named_coords_none() -> NamedCoordsMojo:
    return NamedCoordsMojo()


def _u32_at(base: Int, off: Int) -> UInt32:
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=base + off)
    var b0 = UInt32(p[0])
    var b1 = UInt32(p[1])
    var b2 = UInt32(p[2])
    var b3 = UInt32(p[3])
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)


def named_coords_lookup(
    idx: NamedCoordsMojo, node_id: Int, mut seg: Int, mut so: Int
) raises:
    if not idx.alive or node_id < 1 or node_id > idx.n_nodes:
        raise Error("named_coords lookup OOR node=" + String(node_id))
    var off = node_id * 8
    seg = Int(_u32_at(idx.addr, off))
    so = Int(_u32_at(idx.addr, off + 4))
    if seg == 0:
        raise Error("named_coords empty node=" + String(node_id))


def named_coords_translate_path(
    path: String,
    idx: NamedCoordsMojo,
    mut out_path: String,
    mut path_start: Int,
) raises:
    """Collapse consecutive chopped nodes that tile the same GFA segment."""
    out_path = path
    path_start = 0
    if not idx.alive or path == "*" or path.byte_length() == 0:
        return
    var built = String("")
    var first = True
    var prev_seg = -1
    var i = 0
    var n = path.byte_length()
    while i < n:
        var ch = String(path[byte = i : i + 1])
        if ch != ">" and ch != "<":
            i += 1
            continue
        var orient = ch
        i += 1
        var start = i
        while i < n:
            var d = String(path[byte = i : i + 1])
            if d == ">" or d == "<":
                break
            i += 1
        if i <= start:
            continue
        var nid = Int(String(path[byte = start : i]))
        var seg = 0
        var so = 0
        named_coords_lookup(idx, nid, seg, so)
        if first:
            path_start = so
            first = False
        if prev_seg < 0 or seg != prev_seg:
            built = built + orient + String(seg)
            prev_seg = seg
    if built.byte_length() == 0:
        return
    out_path = built


def named_coords_open_from_dir(index_dir: String) raises -> NamedCoordsMojo:
    var out = NamedCoordsMojo()
    var os_mod = Python.import_module("os")
    var pathlib = Python.import_module("pathlib")
    var root = pathlib.Path(index_dir)
    var nodes = root / "nodes.bin"
    var meta = root / "meta.json"
    if not Bool(nodes.is_file()) or not Bool(meta.is_file()):
        return out^
    var size = Int(py=os_mod.stat(String(nodes)).st_size)
    if size < 16 or size % 8 != 0:
        return out^
    var fd = Int(py=os_mod.open(String(nodes), os_mod.O_RDONLY))
    var hint_byte: UInt8 = 0
    var hint = UnsafePointer(to=hint_byte)
    # PROT_READ=1, MAP_PRIVATE=2
    var addr = external_call["mmap", UnsafePointer[UInt8, MutAnyOrigin]](
        hint, UInt(size), Int32(1), Int32(2), Int32(fd), Int64(0)
    )
    var addr_i = Int(addr)
    if addr_i == -1 or addr_i == 0:
        _ = os_mod.close(fd)
        raise Error("named_coords mmap failed: " + String(nodes))
    out.addr = addr_i
    out.nbytes = size
    out.fd = fd
    out.n_nodes = (size // 8) - 1
    out.alive = True
    print("mojo_gaf named_coords emit-time mmap index=", index_dir, flush=True)
    return out^


def named_coords_try_open() raises -> NamedCoordsMojo:
    """Open mmap index only when METHYLGRAPHER_NAMED_COORDS_INDEX is set.

    The fleet default under mojo_segments is Buffy/HPRC-specific; applying it
    to toy GBZs silently rewrites node ids. Align workers export the env for
    production dual-map.
    """
    var os_mod = Python.import_module("os")
    var env_idx = String(
        os_mod.environ.get("METHYLGRAPHER_NAMED_COORDS_INDEX", "")
    )
    # strip via Python to keep a real String (Mojo String.strip -> StringSlice).
    env_idx = String(env_idx.strip())
    if env_idx.byte_length() == 0:
        return named_coords_none()
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, "/opt/methylgrapher-mojo")
    sys_mod.path.insert(0, "/home/ubuntu/mojo-align/methylgrapher")
    sys_mod.path.insert(0, "/home/ubuntu/mojo-align")
    try:
        var nc = Python.import_module("engine.named_coords")
        if not Bool(nc.index_ready(env_idx)):
            print(
                "mojo_gaf named_coords index not ready: ",
                env_idx,
                flush=True,
            )
            return named_coords_none()
        return named_coords_open_from_dir(env_idx)
    except e:
        print("mojo_gaf named_coords emit-time skipped: ", e, flush=True)
        return named_coords_none()


def named_coords_close(mut idx: NamedCoordsMojo) raises:
    if not idx.alive:
        return
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=idx.addr)
    _ = external_call["munmap", Int32](p, UInt(idx.nbytes))
    var os_mod = Python.import_module("os")
    if idx.fd >= 0:
        try:
            _ = os_mod.close(idx.fd)
        except e:
            pass
    idx.alive = False
    idx.addr = 0
    idx.fd = -1


def named_coords_from_wrap(wrap: PythonObject) raises -> NamedCoordsMojo:
    var idx = NamedCoordsMojo()
    try:
        var addr = Int(py=wrap["named_addr"])
        if addr == 0:
            return idx^
        idx.addr = addr
        idx.nbytes = Int(py=wrap["named_nbytes"])
        idx.fd = Int(py=wrap["named_fd"])
        idx.n_nodes = Int(py=wrap["named_n_nodes"])
        idx.alive = True
    except e:
        pass
    return idx^


def named_coords_store_wrap(wrap: PythonObject, idx: NamedCoordsMojo) raises:
    wrap["named_addr"] = idx.addr
    wrap["named_nbytes"] = idx.nbytes
    wrap["named_fd"] = idx.fd
    wrap["named_n_nodes"] = idx.n_nodes
