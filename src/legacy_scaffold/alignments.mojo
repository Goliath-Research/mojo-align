# alignments.mojo
# Port of alignments.py — alignment entry point, GAF merging,
# and BAM reading via samtools (no pysam dependency).
#
# Migration notes:
#   - pysam.AlignmentFile -> "samtools view -h" subprocess + SAM text parser
#   - multiprocessing workers -> parallelize() stubs (see mgmp.mojo)
#   - GAF format: vg giraffe output, text-based, one alignment per line

from collections import Dict, List
from sys import argv
from .utility import system_execute, is_gzip

# ---------------------------------------------------------------------------
# SAM/BAM record  (minimal fields needed by methylation caller)
# ---------------------------------------------------------------------------

struct SamRecord:
    var query_name:  String
    var flag:        Int
    var ref_name:    String
    var pos:         Int       # 1-based POS
    var mapq:        Int
    var cigar:       String
    var seq:         String
    var qual:        String
    var tags:        Dict[String, String]   # tag -> value string

    fn __init__(inout self):
        self.query_name = ""
        self.flag       = 0
        self.ref_name   = ""
        self.pos        = 0
        self.mapq       = 0
        self.cigar      = ""
        self.seq        = ""
        self.qual       = ""
        self.tags       = Dict[String, String]()

    fn is_unmapped(self) -> Bool:
        return (self.flag & 4) != 0

    fn is_reverse(self) -> Bool:
        return (self.flag & 16) != 0

    fn is_secondary(self) -> Bool:
        return (self.flag & 256) != 0

    fn is_supplementary(self) -> Bool:
        return (self.flag & 2048) != 0

# ---------------------------------------------------------------------------
# SAM text parser (one line -> SamRecord)
# ---------------------------------------------------------------------------

fn parse_sam_line(line: String) raises -> SamRecord:
    """Parse a single SAM alignment line into a SamRecord."""
    var rec = SamRecord()
    if line.startswith("@"):
        raise Error("Header line passed to parse_sam_line")
    var f = line.split("\t")
    if len(f) < 11:
        raise Error("Malformed SAM line: fewer than 11 fields")
    rec.query_name = f[0]
    rec.flag       = int(f[1])
    rec.ref_name   = f[2]
    rec.pos        = int(f[3])
    rec.mapq       = int(f[4])
    rec.cigar      = f[5]
    rec.seq        = f[9]
    rec.qual       = f[10]
    # Parse optional tags (field 11+)
    for i in range(11, len(f)):
        var tag_field = f[i]
        var parts = tag_field.split(":")
        if len(parts) >= 3:
            rec.tags[parts[0]] = parts[2]
    return rec

# ---------------------------------------------------------------------------
# samtools-based BAM reader  (replaces pysam.AlignmentFile)
# ---------------------------------------------------------------------------
# samtools is already installed on the target environment.
# We stream "samtools view -h <bam>" and parse the text output.

struct BamReader:
    var bam_path:  String
    var _lines:    List[String]
    var _cursor:   Int

    fn __init__(inout self, bam_path: String) raises:
        self.bam_path = bam_path
        self._cursor  = 0
        # Stream entire file into memory via samtools view
        # For very large BAMs, swap to line-by-line streaming with subprocess
        var cmd = "samtools view -h " + bam_path
        var stdout_stderr = system_execute(cmd)
        var raw = stdout_stderr[0]
        self._lines = raw.split("\n")

    fn has_next(self) -> Bool:
        return self._cursor < len(self._lines)

    fn next_record(inout self) raises -> SamRecord:
        """Return next non-header SAM record."""
        while self._cursor < len(self._lines):
            var line = self._lines[self._cursor]
            self._cursor += 1
            if len(line) == 0 or line.startswith("@"):
                continue
            return parse_sam_line(line)
        raise Error("BamReader: no more records")

# ---------------------------------------------------------------------------
# GAF record  (vg giraffe output format)
# ---------------------------------------------------------------------------
# GAF columns (tab-separated):
#  0  query name
#  1  query length
#  2  query start
#  3  query end
#  4  strand (+/-)
#  5  path (e.g. >s1<s2>s3)
#  6  path length
#  7  path start
#  8  path end
#  9  residue matches
# 10  alignment block length
# 11  mapq
# 12+ optional tags

struct GafRecord:
    var query_name:     String
    var query_len:      Int
    var query_start:    Int
    var query_end:      Int
    var strand:         String
    var path:           String
    var path_len:       Int
    var path_start:     Int
    var path_end:       Int
    var residue_match:  Int
    var block_len:      Int
    var mapq:           Int
    var seq:            String   # optional; populated if AS tag carries it
    var tags:           Dict[String, String]

    fn __init__(inout self):
        self.query_name    = ""
        self.query_len     = 0
        self.query_start   = 0
        self.query_end     = 0
        self.strand        = "+"
        self.path          = ""
        self.path_len      = 0
        self.path_start    = 0
        self.path_end      = 0
        self.residue_match = 0
        self.block_len     = 0
        self.mapq          = 0
        self.seq           = ""
        self.tags          = Dict[String, String]()

    fn identity(self) -> Float64:
        """Fraction of residue matches over block length."""
        if self.block_len == 0:
            return 0.0
        return Float64(self.residue_match) / Float64(self.block_len) * 100.0

fn parse_gaf_line(line: String) raises -> GafRecord:
    """Parse a single GAF line."""
    var rec = GafRecord()
    var f = line.split("\t")
    if len(f) < 12:
        raise Error("Malformed GAF line: fewer than 12 fields")
    rec.query_name    = f[0]
    rec.query_len     = int(f[1])
    rec.query_start   = int(f[2])
    rec.query_end     = int(f[3])
    rec.strand        = f[4]
    rec.path          = f[5]
    rec.path_len      = int(f[6])
    rec.path_start    = int(f[7])
    rec.path_end      = int(f[8])
    rec.residue_match = int(f[9])
    rec.block_len     = int(f[10])
    rec.mapq          = int(f[11])
    for i in range(12, len(f)):
        var parts = f[i].split(":")
        if len(parts) >= 3:
            rec.tags[parts[0]] = parts[2]
    return rec

# ---------------------------------------------------------------------------
# Alignment main entry point  (mirrors alignments.alignment_main)
# ---------------------------------------------------------------------------

fn alignment_main(
    fq1: String,
    fq2: String,
    work_dir: String,
    index_prefix: String,
    compress: Bool = False,
    thread: Int = 1,
    directional: Bool = True,
    vg_path: String = "vg"
) raises:
    """
    Convert FASTQ files to bisulfite form and run vg giraffe alignment.
    Mirrors alignments.alignment_main() from the Python version.
    """
    # TODO: implement full fastq_converter + vg giraffe orchestration
    # Parallel FASTQ conversion -> vg giraffe per conversion type -> merge GAF
    raise Error("alignment_main: not yet implemented — see TODO in alignments.mojo")

fn alignment_merge_main(work_dir: String, worker_num: Int = 1) raises:
    """Merge split GAF files from parallel alignment runs."""
    # TODO: implement GAF merge logic
    raise Error("alignment_merge_main: not yet implemented")

fn alignment_cleanup(work_dir: String) raises:
    """Remove intermediate alignment files."""
    var cmd = "rm -f " + work_dir + "/C2T.* " + work_dir + "/G2A.*"
    _ = system_execute(cmd)
