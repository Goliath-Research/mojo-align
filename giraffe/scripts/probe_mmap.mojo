# Probe: libc mmap + read u64 from Mojo pointer.

from std.sys import exit
from std.ffi import external_call
from std.memory import UnsafePointer
from std.python import Python


def main() raises:
    var path = "/home/ubuntu/methylGrapher-mojo/tests/data/giraffe_fixture/gbz_toy/toy.wl.C2T.shortread.withzip.min"
    var os_mod = Python.import_module("os")
    var st = os_mod.stat(path)
    var size = Int(py=st.st_size)
    var fd = Int(py=os_mod.open(path, os_mod.O_RDONLY))
    print("fd=", fd, " size=", size)

    # mmap ignores addr unless MAP_FIXED — pass a stack hint (non-null UnsafePointer).
    var hint_byte: UInt8 = 0
    var hint = UnsafePointer(to=hint_byte)
    var addr = external_call["mmap", UnsafePointer[UInt8, MutAnyOrigin]](
        hint, UInt(size), Int32(1), Int32(2), Int32(fd), Int64(0)
    )
    var tag = (
        Int(addr[0])
        | (Int(addr[1]) << 8)
        | (Int(addr[2]) << 16)
        | (Int(addr[3]) << 24)
    )
    print("tag=", tag, " expect=", 0x31513151)
    if tag != 0x31513151:
        print("FAIL tag")
        exit(1)
    var u64p = addr.bitcast[UInt64]()
    print("k=", Int(u64p[1]), " w=", Int(u64p[2]))
    _ = external_call["munmap", Int32](addr, UInt(size))
    _ = os_mod.close(fd)
    print("PASS")
    exit(0)
