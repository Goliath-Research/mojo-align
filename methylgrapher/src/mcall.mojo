# src/mcall.mojo
# Native Mojo MethylCall hot path.
#
# Ports `alignment_to_methylation()` and drives MethylCall with:
#   - native GFA segment Dict (`gfa.GraphicalFragmentAssemblyMemory`)
#   - native per-base methylation calling (this file + `mcall_core`)
#   - Mojo `parallelize()` over fragments within each alignment batch
#
# Alignment filtering / best-alignment selection stays in `engine.mcall`
# (complex GAF bookkeeping); tmp-file merge also reuses the Python merge
# step for exact output parity. See MIGRATION_LOG.md.

from std.algorithm import parallelize
from std.collections import Dict, List
from std.python import Python, PythonObject
from std.sys import exit

from gfa import GraphicalFragmentAssemblyMemory
from mcall_core import AlignmentPath, Indel, alignment_path_parse, cs_tag_parse
from utility import bool_from_str, open_text_write, phred_to_int, reverse_complement


struct MethylCall(Copyable, Movable):
    """One cytosine call: (segmentID, pos, strand, category, methylated)."""

    var segment_id: String
    var segment_pos: Int
    var strand: String
    var category: String
    var methylated: Int

    def __init__(
        out self,
        segment_id: String,
        segment_pos: Int,
        strand: String,
        category: String,
        methylated: Int,
    ):
        self.segment_id = segment_id
        self.segment_pos = segment_pos
        self.strand = strand
        self.category = category
        self.methylated = methylated

    def sort_key(self) -> String:
        return (
            self.segment_id
            + "\t"
            + String(self.segment_pos)
            + "\t"
            + self.strand
            + "\t"
            + self.category
            + "\t"
            + String(self.methylated)
        )


struct GenotypeCall(Copyable, Movable):
    var segment_id: String
    var segment_pos: Int
    var ref_base: String
    var read_base: String
    var phred: String

    def __init__(
        out self,
        segment_id: String,
        segment_pos: Int,
        ref_base: String,
        read_base: String,
        phred: String,
    ):
        self.segment_id = segment_id
        self.segment_pos = segment_pos
        self.ref_base = ref_base
        self.read_base = read_base
        self.phred = phred


struct MethylCallResult(Copyable, Movable):
    var mcalls: List[MethylCall]
    var gcalls: List[GenotypeCall]
    var error_count: Int

    def __init__(out self):
        self.mcalls = List[MethylCall]()
        self.gcalls = List[GenotypeCall]()
        self.error_count = 0


struct ParsedAlignment(Copyable, Movable):
    """Mojo-side view of one best-alignment record for methylation calling."""

    var query_start: Int
    var query_end: Int
    var path: AlignmentPath
    var path_start: Int
    var alignment_tag: String
    var original_bs_read: String
    var read_conversion_type: String
    var phred_score: String
    var tags_ok: Bool

    def __init__(out self):
        self.query_start = 0
        self.query_end = 0
        self.path = AlignmentPath()
        self.path_start = 0
        self.alignment_tag = String("")
        self.original_bs_read = String("")
        self.read_conversion_type = String("")
        self.phred_score = String("")
        self.tags_ok = False


def _char_at(s: String, i: Int) -> String:
    return String(s[byte = i : i + 1])


def _splice_insert_spaces(s: String, start: Int, length: Int) -> String:
    """Insert `length` spaces at `start` (deletion padding on the query)."""
    var spaces = String("")
    for _ in range(length):
        spaces += " "
    return (
        String(s[byte = 0 : start])
        + spaces
        + String(s[byte = start : s.byte_length()])
    )


def _splice_delete(s: String, start: Int, length: Int) -> String:
    """Delete `length` characters starting at `start` (insertion on query)."""
    return String(s[byte = 0 : start]) + String(
        s[byte = start + length : s.byte_length()]
    )


def _complement_base(base: String) -> String:
    if base == "A":
        return "T"
    if base == "T":
        return "A"
    if base == "C":
        return "G"
    if base == "G":
        return "C"
    return "N"


def _sorted_unique_mcalls(calls: List[MethylCall]) raises -> List[MethylCall]:
    """Deduplicate and sort like Python `list(sorted(set(...)))`.

    Pure Mojo (insertion sort on string keys) so this is safe to call from
    `parallelize` workers — must not touch the Python interpreter off-thread.
    """
    var seen = Dict[String, Bool]()
    var keys = List[String]()
    var by_key = Dict[String, MethylCall]()
    for c in calls:
        var k = c.sort_key()
        if k in seen:
            continue
        seen[k] = True
        keys.append(k)
        by_key[k] = c.copy()

    # Insertion sort — fragment call counts are tiny (tens), so O(n^2) is fine.
    var n = len(keys)
    var i = 1
    while i < n:
        var j = i
        while j > 0 and keys[j] < keys[j - 1]:
            var tmp = keys[j].copy()
            keys[j] = keys[j - 1].copy()
            keys[j - 1] = tmp^
            j -= 1
        i += 1

    var out = List[MethylCall]()
    for k in keys:
        out.append(by_key[k].copy())
    return out^


def alignment_to_methylation(
    alignments: List[ParsedAlignment],
    sequence_dict: Dict[String, String],
    cg_only: Bool = True,
    perform_gcall: Bool = False,
    phred_score_threshold: Int = 20,
) raises -> MethylCallResult:
    """Native port of `engine.mcall.alignment_to_methylation`."""
    var result = MethylCallResult()
    var mcall_raw = List[MethylCall]()

    for alignment in alignments:
        if not alignment.tags_ok:
            result.error_count += 1
            continue

        # VG giraffe always outputs + strand in column 4.
        var query_start = alignment.query_start
        var query_end = alignment.query_end
        var path = alignment.path.copy()
        var path_start = alignment.path_start
        var alignment_tag = alignment.alignment_tag.copy()
        var original_bs_read = alignment.original_bs_read.copy()
        var read_conversion_type = alignment.read_conversion_type.copy()
        var phred_score = alignment.phred_score.copy()

        var cs = cs_tag_parse(alignment_tag)
        query_start += cs.query_start_offset
        query_end -= cs.query_end_offset
        var rl = cs.ref_len
        var path_end = path_start + rl

        var path_sequences = List[String]()
        var pi = 0
        while pi < len(path.segments):
            var segmentID = path.segments[pi]
            if segmentID not in sequence_dict:
                raise Error("Missing segment sequence: " + segmentID)
            var seq = sequence_dict[segmentID]
            if path.directions[pi] < 0:
                seq = reverse_complement(seq)
            path_sequences.append(seq)
            pi += 1

        var path_sequence = String("")
        for seq in path_sequences:
            path_sequence += seq

        var path_seq_portion = String(path_sequence[byte = path_start : path_end])
        var bs_read_portion = String(original_bs_read[byte = query_start : query_end])
        var phred_score_portion = String(phred_score[byte = query_start : query_end])

        for indel in cs.indels:
            var indel_start = indel.pos
            var indel_leng = indel.length
            if indel.kind == "-":
                bs_read_portion = _splice_insert_spaces(
                    bs_read_portion, indel_start, indel_leng
                )
                phred_score_portion = _splice_insert_spaces(
                    phred_score_portion, indel_start, indel_leng
                )
            else:
                bs_read_portion = _splice_delete(
                    bs_read_portion, indel_start, indel_leng
                )
                phred_score_portion = _splice_delete(
                    phred_score_portion, indel_start, indel_leng
                )

        if (
            path_seq_portion.byte_length() != rl
            or path_seq_portion.byte_length() != bs_read_portion.byte_length()
            or bs_read_portion.byte_length() != phred_score_portion.byte_length()
        ):
            result.error_count += 1
            continue

        var pl = path_sequence.byte_length()
        var segment_index = 0
        var segmentID = path.segments[segment_index]
        var segment_orientation = path.directions[segment_index]
        var segment_length = path_sequences[segment_index].byte_length()
        var path_len_so_far = segment_length

        var i = 0
        while i < path_seq_portion.byte_length():
            var path_pos = i + path_start
            var ref_base = _char_at(path_seq_portion, i)
            var read_base = _char_at(bs_read_portion, i)
            var phred_score_letter_base = _char_at(phred_score_portion, i)

            if read_base == " ":
                i += 1
                continue

            var phred_score_int_base = phred_to_int(phred_score_letter_base)
            var category = String("U")
            var methylated = 0

            if phred_score_int_base < phred_score_threshold:
                i += 1
                continue

            if ref_base != "C" and ref_base != "G":
                i += 1
                continue

            var interesting_g = False
            var interesting_m = False

            if perform_gcall:
                if ref_base == "C" and read_conversion_type == "G":
                    interesting_g = True
                elif ref_base == "G" and read_conversion_type == "C":
                    interesting_g = True

            if ref_base == "C" and read_conversion_type == "C":
                if read_base == "C" or read_base == "T":
                    interesting_m = True
            elif ref_base == "G" and read_conversion_type == "G":
                if read_base == "G" or read_base == "A":
                    interesting_m = True

            if not interesting_g and not interesting_m:
                i += 1
                continue

            while path_pos >= path_len_so_far:
                segment_index += 1
                segmentID = path.segments[segment_index]
                segment_length = path_sequences[segment_index].byte_length()
                segment_orientation = path.directions[segment_index]
                path_len_so_far += segment_length

            var segment_pos: Int
            if segment_orientation > 0:
                segment_pos = segment_length - (path_len_so_far - path_pos)
            else:
                segment_pos = path_len_so_far - path_pos - 1

            if interesting_g:
                var a = ref_base
                var b = read_base
                if segment_orientation < 0:
                    a = _complement_base(ref_base)
                    b = _complement_base(read_base)
                result.gcalls.append(
                    GenotypeCall(
                        segmentID, segment_pos, a, b, phred_score_letter_base
                    )
                )
                i += 1
                continue

            var base_strand = segment_orientation
            if ref_base == "C":
                if path_pos + 1 < pl:
                    var ref_base2 = _char_at(path_sequence, path_pos + 1)
                    if ref_base2 == "G":
                        category = "CG"
                    else:
                        if path_pos + 2 < pl:
                            var ref_base3 = _char_at(path_sequence, path_pos + 2)
                            if ref_base3 == "G":
                                category = "CHG"
                            else:
                                category = "CHH"
                if read_base == "T":
                    methylated = 0
                elif read_base == "C":
                    methylated = 1
            elif ref_base == "G":
                base_strand = -base_strand
                if path_pos > 0:
                    var ref_base2g = _char_at(path_sequence, path_pos - 1)
                    if ref_base2g == "C":
                        category = "CG"
                    else:
                        if path_pos > 1:
                            var ref_base3g = _char_at(path_sequence, path_pos - 2)
                            if ref_base3g == "C":
                                category = "CHG"
                            else:
                                category = "CHH"
                if read_base == "G":
                    methylated = 1
                elif read_base == "A":
                    methylated = 0
            else:
                i += 1
                continue

            var strand_s = String("+")
            if base_strand < 0:
                strand_s = "-"

            if cg_only and category != "CG":
                i += 1
                continue

            mcall_raw.append(
                MethylCall(segmentID, segment_pos, strand_s, category, methylated)
            )
            i += 1

    result.mcalls = _sorted_unique_mcalls(mcall_raw)
    return result^


def _python_path_to_alignment_path(path_obj: PythonObject) raises -> AlignmentPath:
    var res = AlignmentPath()
    var segs = path_obj[0]
    var dirs = path_obj[1]
    var n = Int(py=Python.import_module("builtins").len(segs))
    for i in range(n):
        res.segments.append(String(segs[i]))
        res.directions.append(Int(py=dirs[i]))
    return res^


def _python_alignment_to_parsed(aln: PythonObject) raises -> ParsedAlignment:
    """Convert one Python best-alignment list into `ParsedAlignment`."""
    var out = ParsedAlignment()
    out.query_start = Int(py=aln[2])
    out.query_end = Int(py=aln[3])
    out.path = _python_path_to_alignment_path(aln[5])
    out.path_start = Int(py=aln[7])

    var builtins = Python.import_module("builtins")
    var n = Int(py=builtins.len(aln))
    var alignment_tag = String("")
    var original_bs_read = String("")
    var read_conversion_type = String("")
    var phred_score = String("")
    var have_cs = False
    var have_os = False
    var have_rc = False
    var have_bq = False

    for ti in range(12, n):
        var tag = String(aln[ti]).strip()
        if tag.startswith("cs:Z:"):
            alignment_tag = String(tag[byte = 5 : tag.byte_length()])
            have_cs = True
        elif tag.startswith("os:Z:"):
            original_bs_read = String(tag[byte = 5 : tag.byte_length()])
            have_os = True
        elif tag.startswith("rc:Z:"):
            read_conversion_type = String(tag[byte = 5 : 6])
            have_rc = True
        elif tag.startswith("bq:Z:"):
            phred_score = String(tag[byte = 5 : tag.byte_length()])
            have_bq = True

    # Mojo GAF may omit bq:Z; synthesize Q40 so MethylCall can proceed.
    if have_os and not have_bq and original_bs_read.byte_length() > 0:
        var n = original_bs_read.byte_length()
        var bq = String("")
        var bi = 0
        while bi < n:
            bq = bq + "I"
            bi += 1
        phred_score = bq
        have_bq = True

    out.alignment_tag = alignment_tag
    out.original_bs_read = original_bs_read
    out.read_conversion_type = read_conversion_type
    out.phred_score = phred_score
    out.tags_ok = have_cs and have_os and have_rc and have_bq
    return out^


def _collect_segment_ids(alignments: List[ParsedAlignment]) -> List[String]:
    var seen = Dict[String, Bool]()
    var out = List[String]()
    for aln in alignments:
        for seg in aln.path.segments:
            if seg in seen:
                continue
            seen[seg] = True
            out.append(seg)
    return out^


def _format_mcalls_for_tmp(calls: List[MethylCall]) -> String:
    """Format calls like the Python worker tmp-file writer (segment-run compressed)."""
    var buf = String("")
    var last_segmentID = String("")
    for c in calls:
        if c.segment_id != last_segmentID:
            buf += (
                c.segment_id
                + "\t"
                + String(c.segment_pos)
                + "\t"
                + c.strand
                + "\t"
                + c.category
                + "\t"
                + String(c.methylated)
                + "\n"
            )
            last_segmentID = c.segment_id
        else:
            buf += (
                "\t"
                + String(c.segment_pos)
                + "\t"
                + c.strand
                + "\t"
                + c.category
                + "\t"
                + String(c.methylated)
                + "\n"
            )
    return buf


def _shard_index_from_segment(segment_id: String, n_shards: Int = 100) -> Int:
    """Match Python worker: `int(segmentID[-2:])` for 100 shards."""
    var bl = segment_id.byte_length()
    if bl == 0:
        return 0
    var start = bl - 2
    if start < 0:
        start = 0
    var tail = segment_id[byte = start : bl]
    try:
        var v = Int(tail)
        if v < 0:
            return 0
        return v % n_shards
    except:
        # Non-numeric IDs (toy uses "1","2"): fall back to last digit / hash.
        var last = segment_id[byte = bl - 1 : bl]
        try:
            return Int(last) % n_shards
        except:
            return 0


def _ensure_repo_on_sys_path() raises:
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    var repo_root = String(os_mod.getcwd())
    sys_mod.path.insert(0, repo_root + "/methylgrapher")
    sys_mod.path.insert(0, repo_root)


def get_kv_value(args: List[String], key: String, default: String) -> String:
    var i = 0
    while i < len(args):
        var a = args[i]
        if a.byte_length() > 1 and a.startswith("-") and a[byte = 1 : a.byte_length()] == key:
            if i + 1 < len(args):
                return args[i + 1]
            return default
        i += 1
    return default


def run_methylcall_native(args: List[String]) raises -> Int:
    """Native MethylCall driver (argv shape matches engine.cli MethylCall)."""
    _ensure_repo_on_sys_path()

    var work_dir = get_kv_value(args, "work_dir", "")
    var index_prefix = get_kv_value(args, "index_prefix", "")
    if work_dir.byte_length() == 0 or index_prefix.byte_length() == 0:
        print("MethylCall requires -work_dir and -index_prefix")
        return 2

    var cg_only = bool_from_str(get_kv_value(args, "cg_only", "Y"))
    var genotyping_cytosine = bool_from_str(
        get_kv_value(args, "genotyping_cytosine", "N")
    )
    var discard_multimapped = bool_from_str(
        get_kv_value(args, "discard_multimapped", "Y")
    )
    var minimum_identity = Int(get_kv_value(args, "minimum_identity", "20"))
    var minimum_mapq = Int(get_kv_value(args, "minimum_mapq", "0"))
    var process_count = Int(get_kv_value(args, "t", "1"))
    var batch_size = Int(get_kv_value(args, "batch_size", "4096"))
    if process_count < 1:
        process_count = 1
    if batch_size < 1:
        batch_size = 4096

    var gfa_fp = index_prefix + ".wl.gfa"
    var node_replacement_dict_fp = index_prefix + ".wl.node.replacement.json"

    print("Native MethylCall: loading GFA into Mojo Dict: " + gfa_fp)
    var gfa_instance = GraphicalFragmentAssemblyMemory()
    gfa_instance.parse(gfa_fp)
    print(
        "Native MethylCall: loaded "
        + String(gfa_instance.segment_count())
        + " segments"
    )

    var mcall_mod = Python.import_module("engine.mcall")
    var builtins = Python.import_module("builtins")
    var batch_iter = mcall_mod.iter_alignment_batches(
        work_dir,
        node_replacement_dict_fp,
        minimum_identity=minimum_identity,
        minimum_mapq=minimum_mapq,
        discard_multimapped=discard_multimapped,
        genotyping_cytosine=genotyping_cytosine,
        batch_size=batch_size,
    )

    # Open 100 shard tmp files for worker 0 (native path uses a single logical
    # writer identity; parallelize is over fragments inside each batch).
    var methylation_outputs = List[PythonObject]()
    for i in range(100):
        var mo = work_dir + "/mcall.0." + String(i) + ".tmp"
        methylation_outputs.append(open_text_write(mo))

    var total_frags = 0
    while True:
        var batch_obj: PythonObject
        try:
            batch_obj = builtins.next(batch_iter)
        except:
            break

        var n_frags = Int(py=builtins.len(batch_obj))
        if n_frags == 0:
            continue

        # Materialize Mojo alignments for the batch.
        var frag_alignments = List[List[ParsedAlignment]]()
        for fi in range(n_frags):
            var best = batch_obj[fi]
            var n_aln = Int(py=builtins.len(best))
            var parsed = List[ParsedAlignment]()
            for ai in range(n_aln):
                parsed.append(_python_alignment_to_parsed(best[ai]))
            frag_alignments.append(parsed^)

        # Per-fragment call results (filled under parallelize).
        var frag_results = List[MethylCallResult]()
        for _ in range(n_frags):
            frag_results.append(MethylCallResult())

        @parameter
        def work(fi: Int):
            try:
                var alns = frag_alignments[fi].copy()
                var segs = _collect_segment_ids(alns)
                var seq_dict = gfa_instance.get_sequences_by_segment_ID(segs)
                frag_results[fi] = alignment_to_methylation(
                    alns,
                    seq_dict,
                    cg_only=cg_only,
                    perform_gcall=genotyping_cytosine,
                )
            except e:
                print("Native MethylCall fragment error:", e)

        parallelize[work](n_frags, process_count)

        # Write shard tmp lines. Always emit the full 5-column form so merge
        # is independent of cross-fragment segment-run compression.
        for fi in range(n_frags):
            var calls = frag_results[fi].mcalls.copy()
            for c in calls:
                var shard = _shard_index_from_segment(c.segment_id, 100)
                var fh = methylation_outputs[shard]
                fh.write(
                    c.segment_id
                    + "\t"
                    + String(c.segment_pos)
                    + "\t"
                    + c.strand
                    + "\t"
                    + c.category
                    + "\t"
                    + String(c.methylated)
                    + "\n"
                )
        total_frags += n_frags

    for fh in methylation_outputs:
        fh.close()

    print(
        "Native MethylCall: processed "
        + String(total_frags)
        + " fragments; merging tmp files"
    )

    var cpu_count = process_count
    if process_count > 10:
        cpu_count = 10
    mcall_mod.extraction_merge_and_cleanup_mp(
        work_dir, cpu_count, max_pid=100, coverage_threshold=5
    )
    if cg_only and genotyping_cytosine:
        mcall_mod.cytosine_CG_validation(index_prefix, work_dir)

    var sys_mod = Python.import_module("sys")
    sys_mod.stdout.flush()
    sys_mod.stderr.flush()
    return 0
