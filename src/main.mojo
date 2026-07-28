# main.mojo
# CLI dispatcher — port of main.py
#
# Commands (same as Python version):
#   preparegenome  — convert GFA + index for vg giraffe
#   align          — FASTQ -> bisulfite conversion -> vg giraffe
#   methylcall     — GAF -> methylation calls
#   conversionrate — estimate bisulfite conversion rate
#   mergecpg       — merge cytosine calls into CpG-level output
#   main           — align + methylcall in sequence
#   mergegaf       — manual GAF merge (low-memory fallback)
#   vg_check       — verify vg binary
#   help / -h      — print usage

from sys import argv, exit
from .utility import (
    ConfigParser, gfa_converter, bool_from_str,
    system_execute, read_graph_methyl
)
from .alignments import alignment_main, alignment_merge_main, alignment_cleanup
from .gfa import (
    GraphicalFragmentAssembly, add_lambda_genome_to_gfa,
    get_all_cpg_from_graph
)
from .mcall import mcall_main

# ---------------------------------------------------------------------------
# Version & help text
# ---------------------------------------------------------------------------

alias VERSION = "0.1.0-mojo"

fn help_text() -> String:
    return """
methylGrapher-mojo v""" + VERSION + """

Usage: mojo src/main.mojo <command> [arguments]

Commands:
  help            Print this message
  PrepareGenome   Add lambda spike-in, convert GFA, build vg index
  Align           Convert FASTQs and run vg giraffe alignment
  MethylCall      Call methylation from GAF alignments
  ConversionRate  Estimate bisulfite conversion rate (requires spike-in)
  MergeCpG        Merge per-cytosine calls into CpG-level output
  Main            Run Align + MethylCall in sequence
  MergeGAF        Manually merge split GAF files (low-memory fallback)
  vg_check        Verify vg binary installation

Arguments (same as Python version):
  -gfa            Input GFA file path
  -prefix         Output file prefix
  -fq1 / -fq2     FASTQ input files
  -index_prefix   Path prefix of vg giraffe index
  -work_dir       Working directory (default: ./)
  -t              Thread count (default: 1)
  -lp             Lambda phage reference FASTA (for PrepareGenome)
  -minimum_identity   Min alignment identity % (default: 20)
  -minimum_mapq       Min mapping quality (default: 0)
  -discard_multimapped  Y/N (default: Y)
  -cg_only          Y/N — output only CG context (default: Y)
  -batch_size       Reads per processing batch (default: 4096)
  -directional      Y/N — directional library (default: Y)
  -compress         Y/N — gzip intermediate files (default: N)
"""

# ---------------------------------------------------------------------------
# Argument parser  (mirrors manual argv parsing in main.py)
# ---------------------------------------------------------------------------

fn parse_args(args: List[String]) raises -> Dict[String, String]:
    """Parse -key value pairs from command-line arguments."""
    var kv = Dict[String, String]()
    var i = 0
    while i < len(args):
        var arg = args[i]
        if arg.startswith("-"):
            var key = arg[1:]
            if i + 1 < len(args):
                kv[key] = args[i + 1]
                i += 2
            else:
                raise Error("Missing value for argument: " + arg)
        else:
            raise Error("Unknown positional argument: " + arg)
    return kv

# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

fn main() raises:
    var args = argv()
    # argv()[0] is the script path; skip it
    var cmd_args = List[String]()
    for i in range(1, len(args)):
        cmd_args.append(str(args[i]))

    # Defaults
    var vg_path  = String("vg")
    var thread   = 1

    # Load config.ini if present
    try:
        var cfg = ConfigParser("config.ini")
        cfg.parse()
        var cfg_vg = cfg.get("default", "vg_path")
        var cfg_t  = cfg.get("default", "thread")
        if len(cfg_vg) > 0: vg_path = cfg_vg
        if len(cfg_t)  > 0:
            try: thread = int(cfg_t)
            except: pass
    except:
        pass

    if len(cmd_args) == 0:
        print(help_text())
        exit(0)

    var command = cmd_args[0].lower()
    var rest = List[String]()
    for i in range(1, len(cmd_args)):
        rest.append(cmd_args[i])

    var valid_commands = [
        "preparegenome", "align", "methylcall", "conversionrate",
        "mergecpg", "help", "-h", "--help", "main", "mergegaf", "vg_check"
    ]
    if command not in valid_commands:
        print("Unknown command: " + command)
        exit(1)

    if command in ["help", "-h", "--help"]:
        print(help_text())
        exit(0)

    var kvargs = parse_args(rest)

    # Override thread from CLI
    if "thread" in kvargs:
        try: thread = int(kvargs["thread"])
        except: pass
        _ = kvargs.pop("thread", "")
    if "t" in kvargs:
        try: thread = int(kvargs["t"])
        except: pass
        _ = kvargs.pop("t", "")
    if "vg_path" in kvargs:
        vg_path = kvargs["vg_path"]

    # Common parameters with defaults
    var work_dir             = kvargs.get("work_dir", "./")
    var minimum_identity     = 20.0
    var minimum_mapq         = 0
    var batch_size           = 4096
    var discard_multimapped  = True
    var cg_only              = True
    var compress             = False
    var directional          = True
    var index_prefix         = kvargs.get("index_prefix", "")
    var fq1                  = kvargs.get("fq1", "")
    var fq2                  = kvargs.get("fq2", "")

    try:
        if "minimum_identity" in kvargs: minimum_identity = Float64(int(kvargs["minimum_identity"]))
        if "minimum_mapq"     in kvargs: minimum_mapq     = int(kvargs["minimum_mapq"])
        if "batch_size"       in kvargs: batch_size       = int(kvargs["batch_size"])
        if "discard_multimapped" in kvargs: discard_multimapped = bool_from_str(kvargs["discard_multimapped"])
        if "cg_only"          in kvargs: cg_only          = bool_from_str(kvargs["cg_only"])
        if "compress"         in kvargs: compress         = bool_from_str(kvargs["compress"])
        if "directional"      in kvargs: directional      = bool_from_str(kvargs["directional"])
    except:
        pass

    # ---- dispatch ----

    if command == "preparegenome":
        var gfa_file   = kvargs.get("gfa", "")
        var prefix     = kvargs.get("prefix", "")
        var lambda_ref = kvargs.get("lp", "")

        var gfa_with_lambda = prefix + ".wl.gfa"
        _ = add_lambda_genome_to_gfa(gfa_file, gfa_with_lambda, lambda_ref)
        get_all_cpg_from_graph(gfa_file, prefix + ".cpg.tsv")
        gfa_converter(gfa_with_lambda, prefix + ".wl", compress=compress)

        # Build vg index for each converted GFA
        for conv in ["C2T", "G2A"]:
            var gfa_conv = prefix + ".wl." + conv + ".gfa"
            var idx      = prefix + ".wl." + conv
            var cmd = (
                vg_path + " autoindex -g " + gfa_conv +
                " -p " + idx + " -w giraffe -t " + str(thread)
            )
            _ = system_execute(cmd)
        exit(0)

    if command == "align":
        alignment_main(
            fq1, fq2, work_dir, index_prefix,
            compress=compress, thread=thread,
            directional=directional, vg_path=vg_path
        )
        exit(0)

    if command == "methylcall":
        var gfa_worker_num = 1
        if thread > 20: gfa_worker_num = 2
        mcall_main(
            work_dir, index_prefix,
            cg_only=cg_only,
            minimum_identity=minimum_identity,
            minimum_mapq=minimum_mapq,
            discard_multimapped=discard_multimapped,
            process_count=thread,
            gfa_worker_num=gfa_worker_num,
            batch_size=batch_size
        )
        exit(0)

    if command == "main":
        var gfa_worker_num = 1
        if thread > 20: gfa_worker_num = 2
        alignment_main(
            fq1, fq2, work_dir, index_prefix,
            compress=compress, thread=thread,
            directional=directional, vg_path=vg_path
        )
        mcall_main(
            work_dir, index_prefix,
            cg_only=cg_only,
            minimum_identity=minimum_identity,
            minimum_mapq=minimum_mapq,
            discard_multimapped=discard_multimapped,
            process_count=thread,
            gfa_worker_num=gfa_worker_num,
            batch_size=batch_size
        )
        exit(0)

    if command == "conversionrate":
        # TODO: port estimate_conversion_rate_print from utility.py
        print("conversionrate: not yet implemented in Mojo port")
        exit(1)

    if command == "mergecpg":
        # TODO: port merge_graph_cytosines from utility.py
        print("mergecpg: not yet implemented in Mojo port")
        exit(1)

    if command == "mergegaf":
        alignment_merge_main(work_dir, worker_num=thread)
        alignment_cleanup(work_dir)
        exit(0)

    if command == "vg_check":
        var out_err = system_execute(vg_path + " version")
        print(out_err[0])
        exit(0)
