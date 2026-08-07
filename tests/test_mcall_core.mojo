# tests/test_mcall_core.mojo
# Unit tests for native mcall_core / methylation helpers.
# Run: `pixi run mojo tests/test_mcall_core.mojo` from the repo root
# (or `cd src && mojo ../tests/test_mcall_core.mojo` with src on the path).

from std.collections import Dict, List
from std.sys import exit

# Resolve sibling src/ modules when invoked as tests/test_*.mojo.
from mcall_core import alignment_path_parse, cs_tag_parse
from mcall import ParsedAlignment, alignment_to_methylation
from gfa import GraphicalFragmentAssemblyMemory
from utility import open_text_read, phred_to_int, reverse_complement, write_text_file


def expect(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("ASSERT FAILED: " + msg)


def test_alignment_path_parse() raises:
    var p = alignment_path_parse(">57658665>57658666<123")
    expect(len(p.segments) == 3, "segment count")
    expect(p.segments[0] == "57658665", "seg0")
    expect(p.segments[2] == "123", "seg2")
    expect(p.directions[0] == 1, "dir0")
    expect(p.directions[2] == -1, "dir2")


def test_cs_tag_parse_plain() raises:
    var r = cs_tag_parse(":20")
    expect(r.ref_len == 20, "ref_len plain")
    expect(r.query_start_offset == 0, "qso")
    expect(r.query_end_offset == 0, "qeo")
    expect(len(r.indels) == 0, "no indels")


def test_cs_tag_parse_indels() raises:
    # Leading insertion soft-clip + match + deletion + match
    var r = cs_tag_parse("+AC:5-TT:3")
    expect(r.query_start_offset == 2, "leading +AC soft-clip")
    expect(r.ref_len == 5 + 2 + 3, "ref spans del")
    expect(len(r.indels) == 1, "one remaining indel (del)")
    expect(r.indels[0].kind == "-", "del kind")
    expect(r.indels[0].length == 2, "del len")


def test_reverse_complement_and_phred() raises:
    expect(reverse_complement("ACGT") == "ACGT", "rc palindrome ACGT")
    expect(reverse_complement("AAGG") == "CCTT", "rc AAGG")
    expect(phred_to_int("I") == 40, "phred I")


def test_gfa_parse_toy() raises:
    var g = GraphicalFragmentAssemblyMemory()
    g.parse("tests/data/toy.wl.gfa")
    expect(g.segment_count() == 3, "toy segment count")
    expect(g.get_sequence_by_segment_ID("1") == "ACGTACGTAC", "seg1 seq")
    var ids = List[String]()
    ids.append("1")
    ids.append("2")
    var d = g.get_sequences_by_segment_ID(ids)
    expect(d["2"] == "GTACGTACGT", "seg2 via batch")


def test_alignment_to_methylation_toy_shape() raises:
    # Mirrors tests/data/work_dir/alignment.gaf read1 (C2T side).
    var aln = ParsedAlignment()
    aln.query_start = 0
    aln.query_end = 20
    aln.path = alignment_path_parse(">1>2")
    aln.path_start = 0
    aln.alignment_tag = ":20"
    aln.original_bs_read = "ACGTACGTACGTACGTACGT"
    aln.read_conversion_type = "C"
    aln.phred_score = "IIIIIIIIIIIIIIIIIIII"
    aln.tags_ok = True

    var seqs = Dict[String, String]()
    seqs["1"] = "ACGTACGTAC"
    seqs["2"] = "GTACGTACGT"

    var alns = List[ParsedAlignment]()
    alns.append(aln^)
    var res = alignment_to_methylation(alns, seqs, cg_only=True)
    expect(len(res.mcalls) > 0, "produced CG calls")
    for c in res.mcalls:
        expect(c.category == "CG", "cg_only")


def test_gzip_roundtrip() raises:
    var path = "/tmp/mg_gzip_roundtrip.txt.gz"
    write_text_file(path, "hello-gzip\n", gzip_out=True)
    var fh = open_text_read(path)
    var text = String(fh.read())
    fh.close()
    expect(text.startswith("hello-gzip"), "gzip readback")


def main() raises:
    test_alignment_path_parse()
    test_cs_tag_parse_plain()
    test_cs_tag_parse_indels()
    test_reverse_complement_and_phred()
    test_gfa_parse_toy()
    test_alignment_to_methylation_toy_shape()
    test_gzip_roundtrip()
    print("ALL TESTS PASSED")
    exit(0)
