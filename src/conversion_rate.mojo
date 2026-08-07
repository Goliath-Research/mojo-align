# src/conversion_rate.mojo
# Native Mojo port of engine.utility.estimate_conversion_rate(_print).

from std.collections import Dict, List
from std.python import Python

from utility import get_kv_value, open_text_read


struct CtxCounts(Copyable, Movable):
    var unmet: Int
    var cov: Int

    def __init__(out self, unmet: Int = 0, cov: Int = 0):
        self.unmet = unmet
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


def _find_lambda_segment_id(index_prefix: String) raises -> String:
    var report_fp = index_prefix + ".prepare.genome.report.txt"
    var os_mod = Python.import_module("os")
    if not Bool(os_mod.path.exists(report_fp)):
        raise Error(
            "Cannot find lambda phage segment id in the index report file. "
            "Did you run PrepareGenome with spike-in genome?"
        )
    var fh = open_text_read(report_fp)
    var prefix = String(
        "Insert lambda phage genome into genome graph as segment: "
    )
    while True:
        var line_obj = fh.readline()
        var line = _strip_line(String(line_obj))
        if line.byte_length() == 0 and String(line_obj).byte_length() == 0:
            break
        if line.startswith(prefix):
            var parts = line.split()
            fh.close()
            return String(parts[len(parts) - 1])
    fh.close()
    raise Error(
        "Cannot find lambda phage segment id in the index report file. "
        "Did you run PrepareGenome with spike-in genome?"
    )


def estimate_conversion_rate(
    index_prefix: String, work_dir: String
) raises -> Dict[String, Float64]:
    var lambda_segment_id = _find_lambda_segment_id(index_prefix)
    var by_ctx = Dict[String, CtxCounts]()
    var graph_methyl_fp = work_dir + "/graph.methyl"
    var fh = open_text_read(graph_methyl_fp)
    while True:
        var line_obj = fh.readline()
        var line = _strip_line(String(line_obj))
        if line.byte_length() == 0 and String(line_obj).byte_length() == 0:
            break
        if line.byte_length() == 0:
            continue
        var parts = line.split()
        if len(parts) < 7:
            continue
        if String(parts[0]) != lambda_segment_id:
            continue
        var context = String(parts[3])
        var unmet = Int(String(parts[4]))
        var cov = Int(String(parts[6]))
        if context not in by_ctx:
            by_ctx[context] = CtxCounts(0, 0)
        var cur = by_ctx[context].copy()
        by_ctx[context] = CtxCounts(cur.unmet + unmet, cur.cov + cov)
    fh.close()

    var res = Dict[String, Float64]()
    var unmet_total = 0
    var cov_total = 0
    var ctx_keys = List[String]()
    for ctx in by_ctx.keys():
        ctx_keys.append(ctx)
    for ctx in ctx_keys:
        var c = by_ctx[ctx].copy()
        if c.cov == 0:
            continue
        res[ctx] = Float64(c.unmet) / Float64(c.cov)
        unmet_total += c.unmet
        cov_total += c.cov
    if cov_total == 0:
        raise Error("No lambda-phage cytosine coverage in graph.methyl")
    res["overall"] = Float64(unmet_total) / Float64(cov_total)
    return res^


def estimate_conversion_rate_print(index_prefix: String, work_dir: String) raises -> String:
    var conversion_rate = estimate_conversion_rate(index_prefix, work_dir)
    var overall = conversion_rate["overall"] * 100.0
    var cg_cr = String("Not available")
    var chg_cr = String("Not available")
    var chh_cr = String("Not available")
    if "CG" in conversion_rate:
        cg_cr = String(conversion_rate["CG"] * 100.0) + "%"
    if "CHG" in conversion_rate:
        chg_cr = String(conversion_rate["CHG"] * 100.0) + "%"
    if "CHH" in conversion_rate:
        chh_cr = String(conversion_rate["CHH"] * 100.0) + "%"
    return (
        "Overall conversion rate: "
        + String(overall)
        + "\nCG context conversion rate: "
        + cg_cr
        + "\nCHG context conversion rate: "
        + chg_cr
        + "\nCHH context conversion rate: "
        + chh_cr
        + "\n"
    )


def run_conversion_rate_native(args: List[String]) raises -> Int:
    var work_dir = get_kv_value(args, "work_dir", "./")
    var index_prefix = get_kv_value(args, "index_prefix", "")
    if index_prefix.byte_length() == 0:
        print("ConversionRate requires -index_prefix")
        return 2
    print(estimate_conversion_rate_print(index_prefix, work_dir))
    return 0
