# src/merge_cpg.mojo
# Native Mojo port of engine.utility.merge_graph_cytosines / MergeCpG CLI.

from std.collections import Dict, List
from std.python import Python

from utility import get_kv_value, open_text_read, open_text_write


struct CpgCounts(Copyable, Movable):
    var met: Int
    var cov: Int

    def __init__(out self, met: Int = 0, cov: Int = 0):
        self.met = met
        self.cov = cov


def _strip_line(line: String) -> String:
    var s = line
    while s.byte_length() > 0:
        var last = String(s[byte = s.byte_length() - 1 : s.byte_length()])
        if last == "\n" or last == "\r":
            s = String(s[byte = 0 : s.byte_length() - 1])
        else:
            break
    return s


def _load_graph_methyl_cg(cytosine_fp: String) raises -> Dict[String, CpgCounts]:
    """Load CG rows; keys are 'segmentID\\tpos'."""
    var res = Dict[String, CpgCounts]()
    var fh = open_text_read(cytosine_fp)
    while True:
        var line_obj = fh.readline()
        var line = _strip_line(String(line_obj))
        if line.byte_length() == 0 and String(line_obj).byte_length() == 0:
            break
        if line.byte_length() == 0:
            continue
        var parts = line.split("\t")
        if len(parts) < 8:
            continue
        if String(parts[3]) != "CG":
            continue
        var key = String(parts[0]) + "\t" + String(parts[1])
        res[key] = CpgCounts(Int(String(parts[5])), Int(String(parts[6])))
    fh.close()
    return res^


def _load_false_positive_cytosines(genotype_fp: String) raises -> Dict[String, Bool]:
    var false_pos = Dict[String, Bool]()
    var os_mod = Python.import_module("os")
    if not Bool(os_mod.path.exists(genotype_fp)):
        return false_pos^
    var fh = open_text_read(genotype_fp)
    while True:
        var line_obj = fh.readline()
        var line = _strip_line(String(line_obj))
        if line.byte_length() == 0 and String(line_obj).byte_length() == 0:
            break
        if line.byte_length() == 0:
            continue
        var parts = line.split("\t")
        if len(parts) < 2:
            continue
        false_pos[String(parts[0]) + "\t" + String(parts[1])] = True
    fh.close()
    return false_pos^


def merge_graph_cytosines(
    cpg_fp: String,
    cytosine_fp: String,
    genotype_cytosine_fp: String,
    out_fp: String,
    full_position: Bool = False,
) raises:
    var cytosine_data = _load_graph_methyl_cg(cytosine_fp)
    var false_positive = _load_false_positive_cytosines(genotype_cytosine_fp)

    var out_fh = open_text_write(out_fp)
    var cpg_fh = open_text_read(cpg_fp)
    while True:
        var line_obj = cpg_fh.readline()
        var line = _strip_line(String(line_obj))
        if line.byte_length() == 0 and String(line_obj).byte_length() == 0:
            break
        if line.byte_length() == 0:
            continue
        var parts = line.split("\t")
        if len(parts) < 5:
            continue
        var cpg_ind = String(parts[0])
        var seg1 = String(parts[1])
        var pos1 = String(parts[2])
        var seg2 = String(parts[3])
        var pos2 = String(parts[4])

        var k1 = seg1 + "\t" + pos1
        var k2 = seg2 + "\t" + pos2
        if k1 in false_positive or k2 in false_positive:
            print("SKIP")
            continue

        var met = 0
        var cov = 0
        if k1 in cytosine_data:
            var c1 = cytosine_data[k1].copy()
            met += c1.met
            cov += c1.cov
        if k2 in cytosine_data:
            var c2 = cytosine_data[k2].copy()
            met += c2.met
            cov += c2.cov
        if cov == 0:
            continue

        if full_position:
            out_fh.write(
                cpg_ind
                + "\t"
                + seg1
                + "\t"
                + pos1
                + "\t"
                + seg2
                + "\t"
                + pos2
                + "\t"
                + String(met)
                + "\t"
                + String(cov)
                + "\n"
            )
        else:
            out_fh.write(cpg_ind + "\t" + String(met) + "\t" + String(cov) + "\n")
    cpg_fh.close()
    out_fh.close()


def run_merge_cpg_native(args: List[String]) raises -> Int:
    var work_dir = get_kv_value(args, "work_dir", "./")
    var index_prefix = get_kv_value(args, "index_prefix", "")
    if index_prefix.byte_length() == 0:
        print("MergeCpG requires -index_prefix")
        return 2

    var cpg_fp = index_prefix + ".cpg.tsv"
    var cytosine_fp = work_dir + "/graph.methyl"
    var genotype_fp = work_dir + "/genotype.info.txt"
    var out_fp = work_dir + "/graph.cpg.tsv"

    print("Native MergeCpG: " + cytosine_fp + " + " + cpg_fp + " -> " + out_fp)
    merge_graph_cytosines(cpg_fp, cytosine_fp, genotype_fp, out_fp, full_position=False)
    return 0
