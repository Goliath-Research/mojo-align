# Mojo-native FASTQ for Giraffe stream map — pigz fd + libc read (linear_fastq).
#
# No Python ``gzip.open`` / ``readline`` on the production path. methylGrapher
# converted FASTQs already carry C2T/G2A bodies; open with convert=0 and parse
# MG headers (``os`` / ``rc``) in the caller.

from std.memory import UnsafePointer

from linear_fastq import (
    FastqPairStream,
    FastqPipe,
    _pipe_line,
    fq_close,
    fq_open,
)


def _ascii_span(buf_addr: Int, off: Int, n: Int) raises -> String:
    if n <= 0:
        return String("")
    # Chunked ``+= chr`` (64-byte pieces) — full-string ``+=`` was O(n^2) on long
    # MG headers (embedded original_seq). No Python here: this runs under
    # ``parallelize`` (sync||prefetch) where ``Python.import_module`` segfaults.
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=buf_addr + off)
    var out = String("")
    var chunk = String("")
    var i = 0
    while i < n:
        chunk += chr(Int(p[i]))
        if chunk.byte_length() >= 64:
            out += chunk
            chunk = String("")
        i += 1
    if chunk.byte_length() > 0:
        out += chunk
    return out^


def giraffe_fq_open(fq1: String, fq2: String) raises -> FastqPairStream:
    """Open R1[/R2] via pigz pipe or plain fd (no BS convert in the pipe)."""
    return fq_open(fq1, fq2, "", "")


def giraffe_fq_close(mut s: FastqPairStream) raises:
    fq_close(s)


def giraffe_fq_read_header_seq(
    mut pipe: FastqPipe, mut name: String, mut seq: String
) raises -> Bool:
    """Read one FASTQ record into ``name`` (no leading @) and ``seq``. False at EOF."""
    var hoff = 0
    var hlen = 0
    if not _pipe_line(pipe, hoff, hlen):
        name = String("")
        seq = String("")
        return False
    var soff = 0
    var slen = 0
    var poff = 0
    var plen = 0
    var qoff = 0
    var qlen = 0
    if not _pipe_line(pipe, soff, slen):
        raise Error("truncated FASTQ (missing SEQ)")
    if not _pipe_line(pipe, poff, plen):
        raise Error("truncated FASTQ (missing +)")
    if not _pipe_line(pipe, qoff, qlen):
        raise Error("truncated FASTQ (missing QUAL)")
    _ = poff
    _ = plen
    _ = qoff
    _ = qlen
    var buf_addr = Int(pipe.buf.unsafe_ptr())
    var n = _ascii_span(buf_addr, hoff, hlen)
    var s = _ascii_span(buf_addr, soff, slen)
    if n.startswith("@"):
        n = String(n[byte = 1 : n.byte_length()])
    name = n^
    seq = s^
    return True
