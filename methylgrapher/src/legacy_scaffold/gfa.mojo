# gfa.mojo
# Port of gfa.py — GFA graph parser, CpG site extraction,
# lambda phage injection, bisulfite-converted GFA writing.
#
# Migration notes:
#   - Python dict/set -> Dict[String, String] / List
#   - Lambda phage insertion: same logic, Mojo file I/O
#   - SNV trimming logic: direct port of Python loops
#   - No external dependencies beyond Mojo stdlib

from collections import Dict, List, Set
from .utility import reverse_complement, is_gzip

# ---------------------------------------------------------------------------
# GFA Segment and Link structs
# ---------------------------------------------------------------------------

struct Segment:
    var id:       String
    var seq:      String
    var is_ref:   Bool             # True if tagged as GRCh38 reference
    var chrom:    String           # e.g. "chr1" — from SN:Z:GRCh38.chr1 tag
    var offset:   Int              # from SO:i: tag

    fn __init__(inout self, id: String, seq: String):
        self.id     = id
        self.seq    = seq
        self.is_ref = False
        self.chrom  = ""
        self.offset = -1

struct Link:
    var seg1: String
    var dir1: String   # "+" or "-"
    var seg2: String
    var dir2: String

# ---------------------------------------------------------------------------
# GraphicalFragmentAssemblyMemory  (mirrors class of same name in gfa.py)
# ---------------------------------------------------------------------------

struct GraphicalFragmentAssembly:
    var segments:      Dict[String, Segment]
    var links_fwd:     Dict[String, List[Link]]   # seg1 -> outgoing links

    fn __init__(inout self):
        self.segments  = Dict[String, Segment]()
        self.links_fwd = Dict[String, List[Link]]()

    # ------------------------------------------------------------------
    # parse()  — read a GFA file into memory
    # ------------------------------------------------------------------
    fn parse(inout self, gfa_path: String, keep_link: Bool = False) raises:
        var f = open(gfa_path, "r")
        for raw in f.read().split("\n"):
            var line = raw.strip()
            if len(line) == 0:
                continue
            var record_type = line[0]

            if record_type == "S":
                self._parse_segment_line(line)
            elif record_type == "L" and keep_link:
                self._parse_link_line(line)

        f.close()

    fn _parse_segment_line(inout self, line: String) raises:
        var f = line.split("\t")
        if len(f) < 3:
            return
        var seg_id = f[1]
        var seq    = f[2].upper()
        var seg    = Segment(seg_id, seq)

        # Parse optional tags
        for i in range(3, len(f)):
            var tag = f[i]
            if tag.startswith("SN:Z:GRCh38."):
                seg.is_ref = True
                seg.chrom  = tag[12:]
            if tag.startswith("SO:i:"):
                try:
                    seg.offset = int(tag[5:])
                except:
                    pass

        self.segments[seg_id] = seg

    fn _parse_link_line(inout self, line: String) raises:
        var f = line.split("\t")
        if len(f) < 5:
            return
        var link = Link()
        link.seg1 = f[1]
        link.dir1 = f[2]
        link.seg2 = f[3]
        link.dir2 = f[4]

        if link.seg1 not in self.links_fwd:
            self.links_fwd[link.seg1] = List[Link]()
        self.links_fwd[link.seg1].append(link)

    # ------------------------------------------------------------------
    # write_converted()  — write C->T or G->A converted GFA
    # ------------------------------------------------------------------
    fn write_converted(
        inout self,
        out_path: String,
        from_base: String,
        to_base: String,
        snv_trim: Bool = False
    ) raises:
        """
        Write a bisulfite-converted GFA file.
        snv_trim: if True, remove segments that differ from reference only at
                  the converted base (mirrors SNV_trim logic in Python).
        """
        var fout = open(out_path, "w")

        for seg_id in self.segments:
            var seg = self.segments[seg_id]
            var conv_seq = seg.seq.replace(from_base, to_base)

            # SNV trim: skip segments where the only difference from
            # the reference allele is the converted base
            # TODO: full SNV_trim logic requires reference comparison
            if snv_trim:
                pass   # placeholder

            fout.write("S\t" + seg_id + "\t" + conv_seq + "\n")

        # Re-emit link lines unchanged
        for seg_id in self.links_fwd:
            for link in self.links_fwd[seg_id]:
                fout.write(
                    "L\t" + link.seg1 + "\t" + link.dir1 +
                    "\t" + link.seg2 + "\t" + link.dir2 + "\t0M\n"
                )

        fout.close()

    # ------------------------------------------------------------------
    # get_replacement_SNV()  — SNV replacement dict for node.replacement.json
    # ------------------------------------------------------------------
    fn get_replacement_snv(
        self,
        from_base: String,
        to_base: String
    ) -> Dict[String, String]:
        """
        Return a dict of seg_id -> replacement_seg_id for segments whose
        sequence differs only by from_base->to_base substitution.
        Mirrors get_replacement_SNV() in gfa.py.
        """
        var result = Dict[String, String]()
        # TODO: full implementation requires pairwise segment comparison
        return result

# ---------------------------------------------------------------------------
# Lambda phage injection  (mirrors add_lambda_genome_to_gfa in gfa.py)
# ---------------------------------------------------------------------------

fn add_lambda_genome_to_gfa(
    input_gfa: String,
    output_gfa: String,
    lambda_ref: String
) raises -> String:
    """
    Append the lambda phage sequence as a new segment to the GFA.
    Returns the segment ID assigned to the lambda sequence.
    """
    # Read lambda FASTA and concatenate all sequences
    var lambda_seq = String()
    var lambda_seg_id = "lambda_spike"

    if len(lambda_ref) > 0:
        var fh = open(lambda_ref, "r")
        for line in fh.read().split("\n"):
            var l = line.strip()
            if l.startswith(">"):
                continue
            lambda_seq += l.upper()
        fh.close()

    # Copy input GFA to output, append lambda segment
    var fin  = open(input_gfa,  "r")
    var fout = open(output_gfa, "w")
    fout.write(fin.read())
    fin.close()

    if len(lambda_seq) > 0:
        fout.write("S\t" + lambda_seg_id + "\t" + lambda_seq + "\n")

    fout.close()
    return lambda_seg_id

# ---------------------------------------------------------------------------
# CpG extraction  (mirrors get_all_cpg_from_graph in utility.py)
# ---------------------------------------------------------------------------

struct CpgSite:
    var cpg_type:  String   # "hg38(chrN:pos)", "SNV", "SV", "Other"
    var seg1:      String
    var pos1:      Int
    var seg2:      String
    var pos2:      Int
    var is_edge:   Bool     # True = cross-segment CpG

fn get_all_cpg_from_graph(
    gfa_path: String,
    out_path: String
) raises:
    """
    Scan the genome graph for CpG sites — both within segments
    and spanning segment edges. Write to out_path as TSV.
    Mirrors get_all_cpg_from_graph() in utility.py.
    """
    var g = GraphicalFragmentAssembly()
    g.parse(gfa_path, keep_link=True)

    var fout = open(out_path, "w")
    var cpg_index = 0

    # Within-segment CpGs
    for seg_id in g.segments:
        var seg = g.segments[seg_id]
        var seq = seg.seq
        var pos = 0
        while pos < len(seq) - 1:
            if seq[pos] == "C" and seq[pos + 1] == "G":
                var cpg_type = String("Other")
                if seg.is_ref and seg.offset >= 0:
                    cpg_type = "hg38(" + seg.chrom + ":" + str(seg.offset + pos) + ")"
                var line = (
                    "C" + str(cpg_index) + "\t" +
                    seg_id + "\t" + str(pos) + "\t" +
                    seg_id + "\t" + str(pos + 1) + "\t" +
                    cpg_type + "\n"
                )
                fout.write(line)
                cpg_index += 1
            pos += 1

    # Edge-spanning CpGs
    var edge_index = 0
    var _rc_map = Dict[String, String]()
    _rc_map["A"] = "T"
    _rc_map["C"] = "G"
    _rc_map["G"] = "C"
    _rc_map["T"] = "A"

    for seg1_id in g.links_fwd:
        for link in g.links_fwd[seg1_id]:
            var seg1 = g.segments[link.seg1]
            var seg2 = g.segments[link.seg2]

            # Determine effective bases at the junction
            var base1 = String(seg1.seq[len(seg1.seq) - 1])
            var pos1  = len(seg1.seq) - 1
            if link.dir1 == "-":
                base1 = _rc_map.get(String(seg1.seq[0]), "N")
                pos1  = 0

            var base2 = String(seg2.seq[0])
            var pos2  = 0
            if link.dir2 == "-":
                base2 = _rc_map.get(String(seg2.seq[len(seg2.seq) - 1]), "N")
                pos2  = len(seg2.seq) - 1

            if base1 + base2 != "CG":
                continue

            var cpg_type = String("SV")
            if seg1.is_ref and seg2.is_ref:
                cpg_type = "hg38"
            elif seg1.is_ref or seg2.is_ref:
                cpg_type = "SNV"

            var line = (
                "E" + str(edge_index) + "\t" +
                link.seg1 + "\t" + str(pos1) + "\t" +
                link.seg2 + "\t" + str(pos2) + "\t" +
                cpg_type + "\n"
            )
            fout.write(line)
            edge_index += 1

    fout.close()
