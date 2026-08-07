# src/utility.mojo
# Native Mojo 1.0 utility helpers for methylGrapher-mojo.
#
# This is a small, native-Mojo counterpart to the hot-path primitives in
# python_reference/utility.py (Phred decoding, sequence complement, boolean
# argument parsing, and subprocess execution). It does NOT reimplement the
# full Python engine (FASTQ conversion, GFA I/O, multiprocessing workers,
# etc.) — that remains in `engine/` and is reached from `src/main.mojo` via
# Python interop. See MIGRATION_LOG.md for the module boundary rationale.
#
# Mojo 1.0.0b2 notes:
#   - `fn` was removed; every function below is `def`.
#   - Strings are UTF-8; byte-position slicing/indexing uses `s[byte=a:b]`.
#   - Standard-library imports must be qualified with the `std.` prefix.

from std.collections import List
from std.python import Python, PythonObject


def phred_to_int(qual_char: String) -> Int:
    """Convert a single Phred+33 quality character to its integer score.

    Mirrors `phred_lookup_int` / `phred_to_int()` in python_reference/utility.py.
    """
    return ord(qual_char) - 33


def reverse_complement(seq: String) -> String:
    """Return the reverse complement of a DNA sequence.

    Mirrors `Utility.seq_reverse_complement()` / `sequence_reverse_complement()`
    in python_reference/utility.py (A<->T, C<->G, anything else -> N).
    """
    var result = String("")
    var i = seq.byte_length()
    while i > 0:
        i -= 1
        var base = seq[byte=i : i + 1]
        if base == "A":
            result += "T"
        elif base == "T":
            result += "A"
        elif base == "C":
            result += "G"
        elif base == "G":
            result += "C"
        else:
            result += "N"
    return result


def bool_from_str(s: String) raises -> Bool:
    """Parse a `Y/N`-style CLI boolean argument.

    Mirrors `Utility.argument_boolean()` in python_reference/utility.py:
    true/t/yes/y -> True, false/f/no/n -> False, anything else raises.
    """
    var low = s.lower()
    if low == "true" or low == "t" or low == "yes" or low == "y":
        return True
    elif low == "false" or low == "f" or low == "no" or low == "n":
        return False
    else:
        raise Error("Invalid boolean argument: " + s)


struct ExecResult(Copyable, Movable):
    """Result of `system_execute()`: process exit code plus captured output."""

    var returncode: Int
    var stdout: String
    var stderr: String

    def __init__(out self, returncode: Int, stdout: String, stderr: String):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def ok(self) -> Bool:
        return self.returncode == 0


def system_execute(cmd: String) raises -> ExecResult:
    """Run a shell command to completion and capture its output.

    Mirrors `utility.SystemExecute` in python_reference/utility.py. Mojo 1.0
    has no native subprocess API yet, so this shells out via Python's
    `subprocess` module through Mojo-Python interop (per the original
    migration's design decision, see README.md "Design Decisions").
    """
    var subprocess = Python.import_module("subprocess")
    var result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    var rc = Int(py=result.returncode)
    var out = String(result.stdout)
    var err = String(result.stderr)
    return ExecResult(rc, out, err)


def _looks_gzipped(path: String) -> Bool:
    """True when `path` ends with a gzip extension (case-insensitive)."""
    var low = path.lower()
    return (
        low.endswith(".gz")
        or low.endswith(".gzip")
        or low.endswith(".GZ")
        or low.endswith(".GZIP")
    )


def open_text_read(path: String) raises -> PythonObject:
    """Open a text file for reading, transparently supporting gzip.

    Mirrors the gzip-aware open helpers in `engine/utility.py`. Returns a
    Python file-like object (caller must `.close()`).
    """
    if _looks_gzipped(path):
        var gzip = Python.import_module("gzip")
        return gzip.open(path, "rt")
    var builtins = Python.import_module("builtins")
    return builtins.open(path, "r")


def open_text_write(path: String, gzip_out: Bool = False) raises -> PythonObject:
    """Open a text file for writing; optionally gzip-compress.

    When `gzip_out` is true (or `path` looks gzipped), writes via
    `gzip.open(..., "wt")`.
    """
    if gzip_out or _looks_gzipped(path):
        var gzip = Python.import_module("gzip")
        return gzip.open(path, "wt")
    var builtins = Python.import_module("builtins")
    return builtins.open(path, "w")


def read_text_file(path: String) raises -> String:
    """Read an entire (optionally gzipped) text file into a String."""
    var fh = open_text_read(path)
    var text = String(fh.read())
    fh.close()
    return text


def write_text_file(path: String, content: String, gzip_out: Bool = False) raises:
    """Write `content` to an (optionally gzipped) text file."""
    var fh = open_text_write(path, gzip_out=gzip_out)
    fh.write(content)
    fh.close()


def get_kv_value(args: List[String], key: String, default: String) -> String:
    """Scan `args` for a `-<key> <value>` pair (methylGrapher argv convention)."""
    var i = 0
    while i < len(args):
        var a = args[i]
        if (
            a.byte_length() > 1
            and a.startswith("-")
            and a[byte = 1 : a.byte_length()] == key
        ):
            if i + 1 < len(args):
                return args[i + 1]
            return default
        i += 1
    return default
