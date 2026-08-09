# src/main.mojo
# methylGrapher-mojo CLI dispatcher — Mojo 1.0 entry point.
#
# Native: help, vg_check, Align, MojoGiraffe, MethylCall, MergeCpG,
# ConversionRate. PrepareGenome / Main / MergeGAF forward to engine.cli via
# Python interop. Set METHYLGRAPHER_MCALL_ENGINE=python to force Align /
# MethylCall / MergeCpG / ConversionRate onto engine.cli as well.
# See MIGRATION_LOG.md / README.md.
#
# Usage (same argv shape as upstream methylGrapher / `engine.cli`):
#   mojo src/main.mojo help
#   mojo src/main.mojo MethylCall -work_dir <dir> -index_prefix <prefix> ...
#
# Mojo 1.0.0b2 notes: `fn` was removed (all `def`); stdlib imports need the
# `std.` prefix; Python interop is `from std.python import Python`.

from std.python import Python
from std.sys import argv as sys_argv, exit

from align import run_align_native
from conversion_rate import run_conversion_rate_native
from giraffe_mapper import run_mojo_giraffe_cli
from mcall import run_methylcall_native
from merge_cpg import run_merge_cpg_native
from utility import system_execute

comptime VERSION = "0.1.0-mojo"

# Kept in sync with engine/cli.py and stock methylGrapher 0.2.0 main.py CLI
# defaults (identity=20, mapq=0). Pipeline does not pass these flags.
comptime DEFAULT_MINIMUM_IDENTITY = 20
comptime DEFAULT_MINIMUM_MAPQ = 0


def header() -> String:
    return String(
        "███╗   ███╗███████╗████████╗██╗  ██╗██╗   ██╗██╗      ██████╗ "
        "██████╗  █████╗ ██████╗ ██╗  ██╗███████╗██████╗ \n"
        "████╗ ████║██╔════╝╚══██╔══╝██║  ██║╚██╗ ██╔╝██║     ██╔════╝ "
        "██╔══██╗██╔══██╗██╔══██╗██║  ██║██╔════╝██╔══██╗\n"
        "██╔████╔██║█████╗     ██║   ███████║ ╚████╔╝ ██║     ██║  ███╗"
        "██████╔╝███████║██████╔╝███████║█████╗  ██████╔╝\n"
        "██║╚██╔╝██║██╔══╝     ██║   ██╔══██║  ╚██╔╝  ██║     ██║   ██║"
        "██╔══██╗██╔══██║██╔═══╝ ██╔══██║██╔══╝  ██╔══██╗\n"
        "██║ ╚═╝ ██║███████╗   ██║   ██║  ██║   ██║   ███████╗╚██████╔╝"
        "██║  ██║██║  ██║██║     ██║  ██║███████╗██║  ██║\n"
        "╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚══════╝ ╚═════╝ "
        "╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝"
    )


def help_text() -> String:
    var identity = String(DEFAULT_MINIMUM_IDENTITY)
    var mapq = String(DEFAULT_MINIMUM_MAPQ)

    var body = String(
        "Usage: mojo src/main.mojo <command> <arguments>\n"
        "Or (via bin/methylGrapher): methylGrapher <command> <arguments>\n"
        "Commands:\n"
        "    help\n"
        "    PrepareGenome\n"
        "    Main\n"
        "    Align\n"
        "    MojoGiraffe\n"
        "    MojoFq2bamMeth\n"
        "    MethylCall\n"
        "    ConversionRate\n"
        "    MergeCpG\n"
        "    vg_check\n"
        "\n"
        "Native Mojo commands:\n"
        "    help / vg_check / Align / MethylCall / MergeCpG / ConversionRate\n"
        "    MojoGiraffe (GFA native; GBZ → streaming quartet_map)\n"
        "    Align uses pluggable map backends (cpu_vg | gpu_giraffe | mojo_giraffe).\n"
        "    MojoFq2bamMeth: orchestrated in engine.fq2bam_meth; map kernel is\n"
        "    native src/linear_mapper.mojo (Clara linear substitute).\n"
        "\n"
        "PrepareGenome / Main / MergeGAF dispatch to the Python engine\n"
        "(engine/cli.py). Set METHYLGRAPHER_MCALL_ENGINE=python for full\n"
        "engine.cli rollback of Align / MethylCall / MergeCpG / ConversionRate.\n"
        "\n"
        "PrepareGenome:\n"
        "    methylGrapher PrepareGenome -gfa <path> -prefix <prefix> [-lp <lambda_fa>] [-t <threads>]\n"
        "\n"
        "Main:\n"
        "    methylGrapher Main   # runs Align then MethylCall in sequence\n"
        "\n"
        "Align:\n"
        "    methylGrapher Align -index_prefix <prefix> -fq1 <fastq> [-fq2 <fastq>] -work_dir <dir>\n"
        "        [-t <threads>] [-directional <Y/N>] [-compress <Y/N>]\n"
        "        [-align_engine <cpu_vg|gpu_giraffe|mojo_giraffe>]  (or METHYLGRAPHER_ALIGN_ENGINE)\n"
        "\n"
        "MojoGiraffe (GFA or GBZ→GAF; GPU seed via device helper):\n"
        "    methylGrapher MojoGiraffe -gfa <gfa> -fq1 <fastq> -out_gaf <path>\n"
        "        [-fq2 <fastq>] [-device auto|cpu|nvidia|amd] [-k <kmer>]\n"
        "    methylGrapher MojoGiraffe -gbz <gbz> -dist <dist> -min <min>\n"
        "        [-zipcodes <zip>] -fq1 <fastq> [-fq2 <fastq>] -out_gaf <path>\n"
        "        [-device auto|cpu|nvidia|amd] [-k <kmer>]\n"
        "\n"
        "MojoFq2bamMeth (native Mojo linear map → BAM; portable device):\n"
        "    methylGrapher MojoFq2bamMeth -fq1 <fastq> -fq2 <fastq> -ref <fa>\n"
        "        -out_bam <bam> -out_qc_dir <dir> -sample_id <id>\n"
        "        [-t <threads>] [-device auto|cpu|nvidia|amd] [-work_dir <dir>]\n"
        "        [-k <kmer>]  (default: 15 / METHYLGRAPHER_LINEAR_K)\n"
        "\n"
        "MethylCall:\n"
        "    methylGrapher MethylCall -work_dir <dir> -index_prefix <prefix>\n"
        "        [-minimum_identity <n>] (default: "
        + identity
        + ")\n"
        "        [-minimum_mapq <n>] (default: "
        + mapq
        + ")\n"
        "        [-discard_multimapped <Y/N>] (default: Y)\n"
        "        [-cg_only <Y/N>] (default: Y)\n"
        "        [-genotyping_cytosine <Y/N>] (default: N)\n"
        "        [-t <threads>] [-batch_size <n>] (default: 4096)\n"
        "\n"
        "ConversionRate:\n"
        "    methylGrapher ConversionRate -index_prefix <prefix> -work_dir <dir>\n"
        "\n"
        "MergeCpG:\n"
        "    methylGrapher MergeCpG -index_prefix <prefix> -work_dir <dir>\n"
        "\n"
        "vg_check:\n"
        "    methylGrapher vg_check [-vg_path <path>]\n"
    )

    return "\n\n" + header() + "\n\nmethylGrapher-mojo\nVersion: " + VERSION + "\n\n" + body + "\n\n"


def get_kv_value(args: List[String], key: String, default: String) -> String:
    """Scan `args` for a `-<key> <value>` pair (mirrors the `-key value`
    convention used throughout methylGrapher's argv parsing)."""
    var i = 0
    while i < len(args):
        var a = args[i]
        if a.byte_length() > 1 and a.startswith("-") and a[byte=1 : a.byte_length()] == key:
            if i + 1 < len(args):
                return args[i + 1]
            return default
        i += 1
    return default


def run_vg_check(args: List[String]) raises:
    """Verify the `vg` binary is installed and reachable.

    Native Mojo port of `utility.vg_binary_check()` in
    python_reference/utility.py. Unlike the Python original, this does not
    consult `config.ini` for a default `vg_path` — pass `-vg_path` explicitly
    if `vg` is not on `PATH`.
    """
    var vg_path = get_kv_value(args, "vg_path", "vg")

    var result = system_execute(vg_path + " version")
    var combined = result.stdout + result.stderr

    print("Checking vg binary @ " + vg_path)
    print(combined)

    if "vg version v" in combined:
        print("Success: vg binary seems to be fine")
    else:
        print("Error: Cannot find vg version")


def dispatch_to_engine(raw_args: List[String]) raises -> Int:
    """Forward the CLI argv (unchanged) to the Python engine's
    `engine.cli.main()` via Mojo-Python interop.

    Adds the current working directory (the repo root — `bin/methylGrapher`
    always `cd`s there first) to `sys.path` so `import engine.cli` resolves
    regardless of how Mojo was invoked.
    """
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")

    var repo_root = os_mod.getcwd()
    sys_mod.path.insert(0, repo_root)

    var cli = Python.import_module("engine.cli")

    var py_args = Python.list()
    for a in raw_args:
        py_args.append(a)

    var rc_obj = cli.main(py_args)

    # The embedded CPython interpreter's stdout is buffered separately from
    # Mojo's `print()`; flush explicitly so engine output is not lost/
    # reordered relative to anything Mojo prints afterwards.
    sys_mod.stdout.flush()
    sys_mod.stderr.flush()

    if rc_obj is None:
        return 0
    return Int(py=rc_obj)


def main() raises:
    var raw = sys_argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))

    if len(args) == 0:
        print("No command specified. Use 'help' for more information.")
        print(help_text())
        exit(0)

    var command = args[0].lower()

    var valid_commands = List[String]()
    valid_commands.append("preparegenome")
    valid_commands.append("align")
    valid_commands.append("mojogiraffe")
    valid_commands.append("mojofq2bammeth")
    valid_commands.append("methylcall")
    valid_commands.append("conversionrate")
    valid_commands.append("mergecpg")
    valid_commands.append("help")
    valid_commands.append("-h")
    valid_commands.append("--help")
    valid_commands.append("main")
    valid_commands.append("mergegaf")
    valid_commands.append("vg_check")

    var known = False
    for c in valid_commands:
        if c == command:
            known = True

    if not known:
        print("Unknown command: " + command)
        exit(1)

    if command == "help" or command == "-h" or command == "--help":
        print(help_text())
        exit(0)

    if command == "vg_check":
        run_vg_check(args)
        exit(0)

    # Native Mojo commands. METHYLGRAPHER_MCALL_ENGINE=python (also honored by
    # the Docker entrypoint before Mojo starts) forces full engine.cli rollback.
    var os_mod = Python.import_module("os")
    var mcall_engine = String(os_mod.environ.get("METHYLGRAPHER_MCALL_ENGINE", "native"))
    var force_python = mcall_engine.lower() == "python"

    if command == "methylcall":
        if force_python:
            exit(dispatch_to_engine(args))
        exit(run_methylcall_native(args))

    if command == "mergecpg":
        if force_python:
            exit(dispatch_to_engine(args))
        exit(run_merge_cpg_native(args))

    if command == "conversionrate":
        if force_python:
            exit(dispatch_to_engine(args))
        exit(run_conversion_rate_native(args))

    if command == "align":
        if force_python:
            exit(dispatch_to_engine(args))
        exit(run_align_native(args))

    if command == "mojogiraffe":
        exit(run_mojo_giraffe_cli(args))

    if command == "mojofq2bammeth":
        # Orchestrator in engine.fq2bam_meth (Mojo linear mapper + QC; BWA fallback).
        var os2 = Python.import_module("os")
        var sys2 = Python.import_module("sys")
        sys2.path.insert(0, os2.getcwd())
        var fq = Python.import_module("engine.fq2bam_meth")
        var py_args = Python.list()
        var i = 1
        while i < len(args):
            py_args.append(args[i])
            i += 1
        exit(Int(py=fq.main(py_args)))

    # PrepareGenome / Main / MergeGAF stay on the Python engine.
    var rc = dispatch_to_engine(args)
    exit(rc)
