# utility.mojo
# Port of utility.py — I/O helpers, Phred tables, sequence utils,
# GFA converter, config parser, system executor.
#
# Migration notes:
#   - Python dict  -> Dict[String, String] / Dict[String, Int]
#   - gzip I/O     -> subprocess "zcat" or "gzip -cd" piped to file reader
#   - multiprocessing.Manager().list() -> parallelize() accumulator (see mgmp.mojo)
#   - re.compile() patterns -> manual parsers below (avoids regex stdlib gap)

from collections import Dict, List
from sys import argv
import os

# ---------------------------------------------------------------------------
# Nucleotide complement
# ---------------------------------------------------------------------------

fn complement(base: String) -> String:
    """Return the Watson-Crick complement of a single base."""
    if base == "A": return "T"
    if base == "T": return "A"
    if base == "C": return "G"
    if base == "G": return "C"
    return "N"

fn reverse_complement(seq: String) -> String:
    """Return the reverse complement of a DNA sequence."""
    var result = String()
    var i = len(seq) - 1
    while i >= 0:
        result += complement(String(seq[i]))
        i -= 1
    return result

# ---------------------------------------------------------------------------
# Phred score lookup tables
# ---------------------------------------------------------------------------
# Precomputed at compile time via @parameter + VariadicList
# ASCII 33..126 covers all standard Phred+33 characters.

fn phred_to_int(qual_char: String) -> Int:
    """Convert a Phred+33 quality character to integer score."""
    # ord() equivalent: use built-in ord on first byte
    return ord(qual_char) - 33

fn phred_to_prob(qual_char: String) -> Float64:
    """Convert a Phred+33 quality character to error probability."""
    var q = phred_to_int(qual_char)
    return 10.0 ** (-Float64(q) / 10.0)

# ---------------------------------------------------------------------------
# Config parser  (mirrors ConfigParser class in utility.py)
# ---------------------------------------------------------------------------

struct ConfigParser:
    var _data: Dict[String, Dict[String, String]]
    var _path: String

    fn __init__(inout self, path: String):
        self._path = path
        self._data = Dict[String, Dict[String, String]]()
        self._find()
        # NOTE: call self.parse() after construction

    fn _find(inout self):
        """Search standard locations for config.ini."""
        # TODO: replace with Mojo Path API once stable
        # For now fall through — caller provides explicit path
        pass

    fn parse(inout self) raises:
        var f = open(self._path, "r")
        var block = String()
        for line in f.read().split("\n"):
            var l = line.strip()
            if len(l) == 0 or l.startswith("#"):
                continue
            if l.startswith("["):
                block = l[1 : len(l) - 1]
                self._data[block] = Dict[String, String]()
            elif "=" in l:
                var parts = l.split("=")
                var key = parts[0].strip()
                var val = parts[1].strip()
                self._data[block][key] = val
        f.close()

    fn get(self, block: String, key: String) -> String:
        try:
            return self._data[block][key]
        except:
            return ""

# ---------------------------------------------------------------------------
# System executor  (mirrors SystemExecute in utility.py)
# ---------------------------------------------------------------------------
# Mojo subprocess via os.run() or Python interop.
# Using Python interop for subprocess until Mojo stdlib subprocess matures.

fn system_execute(cmd: String) raises -> Tuple[String, String]:
    """
    Execute a shell command and return (stdout, stderr) as strings.
    Uses Python subprocess under the hood via Mojo-Python interop.
    """
    from python import Python
    var subprocess = Python.import_module("subprocess")
    var result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    return (str(result.stdout), str(result.stderr))

# ---------------------------------------------------------------------------
# GFA base converter  (mirrors gfa_converter() in utility.py)
# ---------------------------------------------------------------------------
# Reads GFA line by line; for S (segment) lines replaces from_base -> to_base
# in the sequence field. No gzip write yet — compress via pipe if needed.

fn gfa_converter(
    input_gfa: String,
    output_prefix: String,
    compress: Bool = False
) raises:
    """Convert a GFA file to C2T and G2A bisulfite forms."""
    var conversions = List[Tuple[String, String, String]]()
    conversions.append(("C2T", "C", "T"))
    conversions.append(("G2A", "G", "A"))

    for conv in conversions:
        var tag = conv[0]
        var from_base = conv[1]
        var to_base = conv[2]
        var out_path = output_prefix + "." + tag + ".gfa"

        # TODO: add gzip support via subprocess "gzip" when compress=True
        var fin = open(input_gfa, "r")
        var fout = open(out_path, "w")

        for raw_line in fin.read().split("\n"):
            var line = raw_line
            if line.startswith("S\t"):
                var fields = line.split("\t")
                if len(fields) >= 3:
                    # Replace only in the sequence column (index 2)
                    fields[2] = fields[2].upper().replace(from_base, to_base)
                    line = "\t".join(fields)
            fout.write(line + "\n")

        fin.close()
        fout.close()

# ---------------------------------------------------------------------------
# FASTQ converter  (mirrors fastq_converter_worker_function in utility.py)
# ---------------------------------------------------------------------------
# Reads FASTQ (4-line records), converts seq bases, embeds original seq
# in the read name exactly as the Python version does.

fn fastq_converter_worker(
    input_fq: String,
    output_fq: String,
    conversion_str: String,   # "C2T" or "G2A"
    split_num: Int = 1000
) raises -> Int:
    """
    Convert a FASTQ file for bisulfite pseudo-alignment.
    Returns number of reads processed.
    """
    var parts = conversion_str.split("2")
    var from_base = parts[0]
    var to_base   = parts[1]

    var fin  = open(input_fq,  "r")
    var fout = open(output_fq, "w")

    var lines = fin.read().split("\n")
    fin.close()

    var i = 0
    var read_count = 0
    while i + 3 < len(lines):
        var header  = lines[i]     # @readname ...
        var seq     = lines[i + 1]
        # lines[i+2] is '+'
        var qual    = lines[i + 3].upper()

        # Parse read name (split on space, take first token)
        var qname = header
        if " " in header:
            qname = header.split(" ")[0]
        else:
            qname = header

        var reminder = read_count % split_num
        var conv_seq = seq.replace(from_base, to_base)
        var new_header = qname + "_" + conversion_str + "_" + str(reminder) + "_" + seq
        fout.write(new_header + "\n" + conv_seq + "\n+\n" + qual + "\n")

        i += 4
        read_count += 1

    fout.close()
    return read_count

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

fn is_gzip(path: String) -> Bool:
    for ext in ["gz", "gzip", "GZ", "GZIP"]:
        if path.endswith("." + ext):
            return True
    return False

fn bool_from_str(s: String) raises -> Bool:
    var low = s.lower()
    if low == "true" or low == "t" or low == "yes" or low == "y":
        return True
    if low == "false" or low == "f" or low == "no" or low == "n":
        return False
    raise Error("Invalid boolean string: " + s)

# ---------------------------------------------------------------------------
# graph_methyl cytosine reader
# ---------------------------------------------------------------------------

struct CytosineMethyl:
    var met: Int
    var cov: Int

fn read_graph_methyl(fp: String) raises -> Dict[String, Dict[String, CytosineMethyl]]:
    """
    Read graph.methyl output: segID -> pos -> (met, cov).
    Only CG context entries are retained.
    """
    var res = Dict[String, Dict[String, CytosineMethyl]]()
    var f = open(fp, "r")
    for raw in f.read().split("\n"):
        var l = raw.strip()
        if len(l) == 0:
            continue
        var fields = l.split("\t")
        if len(fields) < 8:
            continue
        var seg_id  = fields[0]
        var pos     = fields[1]
        var context = fields[3]
        if context != "CG":
            continue
        var met = int(fields[5])
        var cov = int(fields[6])
        if seg_id not in res:
            res[seg_id] = Dict[String, CytosineMethyl]()
        res[seg_id][pos] = CytosineMethyl(met, cov)
    f.close()
    return res
