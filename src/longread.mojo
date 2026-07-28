# longread.mojo
# Port of longread.py — ONT / PacBio long-read methylation calling.
#
# Key difference from short-read path:
#   - Reads Modkit / MM+ML SAM tags for base modification calls
#   - MM tag format: "C+m,0,1,0;" — modification type + skip counts
#   - ML tag: uint8 array of probabilities (0-255 -> 0.0-1.0)
#   - GAF paths for graph-aware long-read alignment
#
# Migration notes:
#   - Python re.findall() for MM tag -> manual state machine parser below
#   - pysam tag access -> samtools view + SAM text parser (see alignments.mojo)
#   - Python list comprehensions -> Mojo List operations

from collections import Dict, List
from .alignments import SamRecord, BamReader, GafRecord, parse_gaf_line
from .gfa import GraphicalFragmentAssembly
from .mcall import CytosineCall, get_cytosine_context, passes_filters

# ---------------------------------------------------------------------------
# MM/ML tag parser  (SAM base modification tags per SAMspec §1.7)
# ---------------------------------------------------------------------------
# MM tag example:  C+m,0,1,0;   means:
#   base C, modification m (5mC), skip counts: 0 then 1 then 0
# ML tag:          uint8 list of probabilities, same order as MM positions

struct BaseModification:
    var query_pos:  Int       # 0-based position in the query sequence
    var prob:       Float64   # probability of modification (0.0 - 1.0)
    var mod_type:   String    # "m" = 5mC, "h" = 5hmC, etc.

fn parse_mm_ml_tags(
    seq: String,
    mm_tag: String,
    ml_tag: String
) -> List[BaseModification]:
    """
    Parse MM and ML SAM tags and return a list of BaseModification entries.
    Implements the skip-count encoding defined in the SAM specification.
    """
    var result = List[BaseModification]()

    if len(mm_tag) == 0:
        return result

    # Split MM tag into modification blocks separated by ";"
    var blocks = mm_tag.split(";")

    # Parse ML probabilities (comma-separated uint8 values)
    var probs = List[Float64]()
    if len(ml_tag) > 0:
        for p_str in ml_tag.split(","):
            try:
                var p = int(p_str.strip())
                probs.append(Float64(p) / 255.0)
            except:
                pass

    var prob_idx = 0

    for block in blocks:
        var block_str = block.strip()
        if len(block_str) == 0:
            continue

        # First field: "C+m" or "C-m" or "C+m?"
        var parts = block_str.split(",")
        if len(parts) < 2:
            continue

        var header = parts[0]   # e.g. "C+m"
        var target_base = String(header[0])  # "C", "A", etc.
        var mod_type = header[2:] if len(header) > 2 else "m"
        # Remove trailing '.' or '?' strand indicators if present
        if mod_type.endswith(".") or mod_type.endswith("?"):
            mod_type = mod_type[:-1]

        # Walk the query sequence, counting target bases, applying skip counts
        var skip_counts = List[Int]()
        for i in range(1, len(parts)):
            try:
                skip_counts.append(int(parts[i].strip()))
            except:
                pass

        var base_count = -1   # count of target bases seen so far
        var skip_idx   = 0

        for qi in range(len(seq)):
            if String(seq[qi]) == target_base:
                base_count += 1
                if skip_idx < len(skip_counts) and base_count == skip_counts[skip_idx]:
                    # This base is modified
                    var prob = probs[prob_idx] if prob_idx < len(probs) else 0.0
                    var bm = BaseModification()
                    bm.query_pos = qi
                    bm.prob      = prob
                    bm.mod_type  = mod_type
                    result.append(bm)
                    prob_idx  += 1
                    skip_idx  += 1
                    base_count = -1   # reset between skips

    return result

# ---------------------------------------------------------------------------
# Methylation probability threshold
# ---------------------------------------------------------------------------
# Mirrors the threshold used in longread.py for calling methylated vs not.

alias METHYLATION_THRESHOLD: Float64 = 0.5

fn is_methylated_lr(prob: Float64) -> Bool:
    return prob >= METHYLATION_THRESHOLD

# ---------------------------------------------------------------------------
# Long-read methylation calling on a single GAF record
# ---------------------------------------------------------------------------

fn call_methylation_longread(
    rec: GafRecord,
    mm_tag: String,
    ml_tag: String,
    graph: GraphicalFragmentAssembly,
    cg_only: Bool,
    inout calls: Dict[String, Dict[Int, CytosineCall]]
) raises:
    """
    Call methylation from a long-read GAF record using MM/ML tags.
    Populates `calls` accumulator the same way as the short-read path.
    """
    var seq = rec.seq
    var mods = parse_mm_ml_tags(seq, mm_tag, ml_tag)

    # Build a quick lookup: query_pos -> probability
    var mod_lookup = Dict[Int, Float64]()
    for bm in mods:
        mod_lookup[bm.query_pos] = bm.prob

    # Walk path segments (same traversal as mcall.mojo)
    from .mcall import parse_gaf_path, _rc
    var segments = parse_gaf_path(rec.path)
    var read_pos = rec.query_start

    for path_seg in segments:
        if path_seg.seg_id not in graph.segments:
            continue
        var seg = graph.segments[path_seg.seg_id]
        var seg_seq = seg.seq if path_seg.is_fwd else _rc(seg.seq)

        for seg_pos in range(len(seg_seq)):
            if read_pos >= len(seq):
                break
            var graph_base = String(seg_seq[seg_pos])

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
                        key_seg, seg_pos,
                        "+" if path_seg.is_fwd else "-",
                        context
                    )

                var prob = mod_lookup.get(read_pos, 0.0)
                calls[key_seg][seg_pos].update(is_methylated_lr(prob))

            read_pos += 1

# ---------------------------------------------------------------------------
# Long-read main entry point  (mirrors mainL.py)
# ---------------------------------------------------------------------------

fn mcall_longread_main(
    work_dir: String,
    index_prefix: String,
    bam_path: String,
    cg_only: Bool = True,
    minimum_mapq: Int = 0,
    process_count: Int = 1
) raises:
    """
    Main entry for long-read methylation calling from a sorted BAM file.
    Uses samtools to stream records; parses MM/ML tags for 5mC probabilities.
    """
    var gfa_path = index_prefix + ".wl.gfa"
    var graph = GraphicalFragmentAssembly()
    graph.parse(gfa_path, keep_link=True)

    var reader = BamReader(bam_path)
    var calls = Dict[String, Dict[Int, CytosineCall]]()

    # TODO Phase 6: batch reads and call with parallelize()
    while reader.has_next():
        try:
            var rec_sam = reader.next_record()
            if rec_sam.is_unmapped() or rec_sam.is_secondary():
                continue
            if rec_sam.mapq < minimum_mapq:
                continue

            var mm_tag = rec_sam.tags.get("MM", "")
            var ml_tag = rec_sam.tags.get("ML", "")

            # Build a minimal GafRecord from the SAM record for path walking
            # TODO: proper SAM->GAF path reconstruction from CIGAR + graph coords
            # For now, emit a warning and skip records without path info
            if "cg" not in rec_sam.tags and "pa" not in rec_sam.tags:
                continue

            # Placeholder: full SAM->graph coordinate mapping needed
            # (requires cs/cg CIGAR tag parsing + segment offset table)
            pass

        except:
            continue

    # Write output
    var out_path = work_dir + "/graph.methyl.longread"
    var fout = open(out_path, "w")
    for seg_id in calls:
        for pos in calls[seg_id]:
            fout.write(calls[seg_id][pos].to_tsv())
    fout.close()
