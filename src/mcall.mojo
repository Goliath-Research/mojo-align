# mcall.mojo
# Port of mcall.py — the methylation calling engine.
# This is the performance-critical hot path of methylGrapher.
#
# Migration strategy:
#   1. Data structures: Python dicts/lists -> Mojo Dict/List structs
#   2. CpG position scan: Python loop -> SIMD-vectorized scan (TODO Phase 4)
#   3. Worker pool: multiprocessing.Queue -> parallelize() (see mgmp.mojo)
#   4. GAF parsing: uses GafRecord from alignments.mojo
#   5. GFA graph access: uses GraphicalFragmentAssembly from gfa.mojo
#
# Performance targets (vs Python baseline):
#   - CpG scanning:       SIMD uint8x16 -> ~8-16x speedup
#   - Batch processing:   no GIL -> near-linear scaling with thread count
#   - Memory:             struct layout vs Python object overhead

from collections import Dict, List
from algorithm import parallelize
from .alignments import GafRecord, parse_gaf_line
from .gfa import GraphicalFragmentAssembly
from .utility import system_execute, bool_from_str

# ---------------------------------------------------------------------------
# Cytosine call record  (one row of graph.methyl output)
# ---------------------------------------------------------------------------

struct CytosineCall:
    var seg_id:   String
    var pos:      Int
    var strand:   String    # "+" or "-"
    var context:  String    # "CG", "CHG", "CHH"
    var unmet:    Int       # unmethylated count
    var met:      Int       # methylated count
    var cov:      Int       # total coverage = met + unmet
    var ml:       Float64   # methylation level = met / cov

    fn __init__(inout self, seg_id: String, pos: Int, strand: String, context: String):
        self.seg_id  = seg_id
        self.pos     = pos
        self.strand  = strand
        self.context = context
        self.unmet   = 0
        self.met     = 0
        self.cov     = 0
        self.ml      = 0.0

    fn update(inout self, is_methylated: Bool):
        self.cov += 1
        if is_methylated:
            self.met += 1
        else:
            self.unmet += 1
        if self.cov > 0:
            self.ml = Float64(self.met) / Float64(self.cov)

    fn to_tsv(self) -> String:
        return (
            self.seg_id + "\t" + str(self.pos) + "\t" +
            self.strand + "\t" + self.context + "\t" +
            str(self.unmet) + "\t" + str(self.met) + "\t" +
            str(self.cov) + "\t" + str(self.ml) + "\n"
        )

# ---------------------------------------------------------------------------
# CpG context determination from a graph segment sequence
# ---------------------------------------------------------------------------
# TODO Phase 4: replace Python-style character loop with SIMD scan
# using SIMD[DType.uint8, 16] for ~8-16x throughput on the hot path.

fn get_cytosine_context(seq: String, pos: Int) -> String:
    """
    Return the methylation context (CG, CHG, CHH) for a cytosine at
    position `pos` in `seq`.  Mirrors the context logic in mcall.py.
    """
    if pos >= len(seq):
        return "Unknown"
    if seq[pos] != "C":
        return "Unknown"

    var next1 = String("N") if pos + 1 >= len(seq) else String(seq[pos + 1])
    var next2 = String("N") if pos + 2 >= len(seq) else String(seq[pos + 2])

    if next1 == "G":
        return "CG"
    # CHG: C followed by any non-G then G
    if next1 != "G" and next2 == "G":
        return "CHG"
    return "CHH"

# ---------------------------------------------------------------------------
# Alignment identity filter  (mirrors identity filter in mcall.py)
# ---------------------------------------------------------------------------

fn passes_filters(
    rec: GafRecord,
    min_identity: Float64,
    min_mapq: Int,
    discard_multimapped: Bool
) -> Bool:
    if rec.mapq < min_mapq:
        return False
    if rec.identity() < min_identity:
        return False
    if discard_multimapped and rec.mapq == 0:
        return False
    return True

# ---------------------------------------------------------------------------
# Path segment extractor  (parse vg GAF path string)
# ---------------------------------------------------------------------------
# GAF path format:  >seg1<seg2>seg3 ...
#   '>' = forward orientation, '<' = reverse complement

struct PathSegment:
    var seg_id:    String
    var is_fwd:    Bool

fn parse_gaf_path(path: String) -> List[PathSegment]:
    """Parse a GAF path string into an ordered list of segments."""
    var result = List[PathSegment]()
    var current_id = String()
    var current_fwd = True

    for i in range(len(path)):
        var ch = String(path[i])
        if ch == ">" or ch == "<":
            if len(current_id) > 0:
                var ps = PathSegment()
                ps.seg_id = current_id
                ps.is_fwd = current_fwd
                result.append(ps)
                current_id = String()
            current_fwd = (ch == ">")
        else:
            current_id += ch

    if len(current_id) > 0:
        var ps = PathSegment()
        ps.seg_id = current_id
        ps.is_fwd = current_fwd
        result.append(ps)

    return result

# ---------------------------------------------------------------------------
# Single-read methylation caller  (core logic of mcall.py)
# ---------------------------------------------------------------------------
# Given one GAF record and the GFA graph, emit CytosineCall updates.
# This function is designed to be called inside parallelize() workers.

fn call_methylation_on_read(
    rec: GafRecord,
    graph: GraphicalFragmentAssembly,
    cg_only: Bool,
    inout calls: Dict[String, Dict[Int, CytosineCall]]
) raises:
    """
    Walk the aligned path of one read and call methylation at each cytosine.
    Updates the `calls` accumulator: seg_id -> pos -> CytosineCall.
    """
    var segments = parse_gaf_path(rec.path)
    var read_seq = rec.seq
    var read_pos = rec.query_start

    for path_seg in segments:
        if path_seg.seg_id not in graph.segments:
            continue
        var seg = graph.segments[path_seg.seg_id]
        var seg_seq = seg.seq if path_seg.is_fwd else _rc(seg.seq)

        for seg_pos in range(len(seg_seq)):
            if read_pos >= len(read_seq):
                break
            var graph_base = String(seg_seq[seg_pos])
            var read_base  = String(read_seq[read_pos])

            # Methylation call logic:
            # In C2T converted reads, unmethylated C appears as T,
            # methylated C appears as C.
            if graph_base == "C":
                var context = get_cytosine_context(seg_seq, seg_pos)
                if cg_only and context != "CG":
                    read_pos += 1
                    continue

                var key_seg = path_seg.seg_id
                if key_seg not in calls:
                    calls[key_seg] = Dict[Int, CytosineCall]()
                if seg_pos not in calls[key_seg]:
                    calls[key_seg][seg_pos] = CytosineCall(
                        key_seg, seg_pos, "+" if path_seg.is_fwd else "-", context
                    )

                var is_met = (read_base == "C")   # T = unmethylated, C = methylated
                calls[key_seg][seg_pos].update(is_met)

            read_pos += 1

fn _rc(seq: String) -> String:
    """Reverse complement helper (local alias)."""
    from .utility import reverse_complement
    return reverse_complement(seq)

# ---------------------------------------------------------------------------
# mcall_main  (mirrors mcall.mcall_main in Python)
# ---------------------------------------------------------------------------

fn mcall_main(
    work_dir: String,
    index_prefix: String,
    cg_only: Bool = True,
    genotyping_cytosine: Bool = False,
    minimum_identity: Float64 = 20.0,
    minimum_mapq: Int = 0,
    discard_multimapped: Bool = True,
    process_count: Int = 1,
    alignment_parse_worker_num: Int = 1,
    gfa_worker_num: Int = 1,
    batch_size: Int = 4096
) raises:
    """
    Main methylation calling function.
    Reads GAF alignments from work_dir, loads GFA graph, emits graph.methyl.
    """
    # Load GFA graph
    var gfa_path = index_prefix + ".wl.gfa"
    var graph = GraphicalFragmentAssembly()
    graph.parse(gfa_path, keep_link=True)

    # Output file
    var out_path = work_dir + "/graph.methyl"
    var fout = open(out_path, "w")

    # Accumulator: seg_id -> pos -> CytosineCall
    var calls = Dict[String, Dict[Int, CytosineCall]]()

    # Process GAF files  (C2T.R1.gaf, G2A.R1.gaf, ...)
    for conv in ["C2T", "G2A"]:
        for read_num in [1, 2]:
            var gaf_path = work_dir + "/" + conv + ".R" + str(read_num) + ".gaf"
            # TODO: check file exists before opening
            var f = open(gaf_path, "r")
            var lines = f.read().split("\n")
            f.close()

            # TODO Phase 6: replace this loop with parallelize() batches
            for raw_line in lines:
                var line = raw_line.strip()
                if len(line) == 0:
                    continue
                try:
                    var rec = parse_gaf_line(line)
                    if not passes_filters(
                        rec, minimum_identity, minimum_mapq, discard_multimapped
                    ):
                        continue
                    call_methylation_on_read(rec, graph, cg_only, calls)
                except:
                    continue

    # Write output
    for seg_id in calls:
        for pos in calls[seg_id]:
            fout.write(calls[seg_id][pos].to_tsv())

    fout.close()
