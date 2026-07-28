# src/mcall_core.mojo
# Native Mojo ports of the two hottest, purely-textual parsers from
# python_reference/mcall.py: `alignment_path_parse()` and `cs_tag_parse()`.
#
# These are the per-alignment-line parsers methylGrapher calls once for
# every GAF record (millions of times on real data), so they are the
# highest-value functions to port natively rather than reach into
# `engine.mcall` over Python interop. The rest of the methylation-calling
# pipeline (multiprocessing workers, GFA sequence lookups, tmp-file
# merging, ...) stays in `engine/` for now — see MIGRATION_LOG.md.


struct AlignmentPath(Copyable, Movable):
    """Result of `alignment_path_parse()`: parallel segment/direction lists.

    Mirrors the `[[segment_ids...], [directions...]]` pair returned by
    `alignment_path_parse()` in python_reference/mcall.py, where
    direction is `1` for `>` (forward) and `-1` for `<` (reverse).
    """

    var segments: List[String]
    var directions: List[Int]

    def __init__(out self):
        self.segments = List[String]()
        self.directions = List[Int]()


def alignment_path_parse(path: String) raises -> AlignmentPath:
    """Parse a GAF path string like `>57658665>57658666<123` into segment IDs
    and per-segment orientations.

    Faithful native port of `alignment_path_parse()` in
    python_reference/mcall.py.
    """
    var res = AlignmentPath()
    var element = String("")

    for ch in path.codepoints():
        var c = String(ch)
        if c == ">" or c == "<":
            if c == ">":
                res.directions.append(1)
            else:
                res.directions.append(-1)

            if element.byte_length() != 0:
                res.segments.append(element)
                element = String("")
        else:
            element += c

    res.segments.append(element)

    if len(res.segments) != len(res.directions):
        raise Error("alignment_path_parse: segment/direction count mismatch in path: " + path)
    for s in res.segments:
        if s.byte_length() == 0:
            raise Error("alignment_path_parse: empty segment ID in path: " + path)

    return res^


struct Indel(Copyable, Movable):
    """A single insertion/deletion event within a cs-tag alignment.

    `pos` is the reference-relative offset (into the aligned reference
    block) at which the indel occurs; `length` is the indel length in
    bases; `kind` is `"+"` (insertion, extra query bases) or `"-"`
    (deletion, extra reference bases) — mirrors the 3-tuples appended to
    `indels` in `cs_tag_parse()` in python_reference/mcall.py.
    """

    var pos: Int
    var length: Int
    var kind: String

    def __init__(out self, pos: Int, length: Int, kind: String):
        self.pos = pos
        self.length = length
        self.kind = kind


struct CsTagResult(Copyable, Movable):
    """Result of `cs_tag_parse()`.

    Mirrors the `(query_start_offset, query_end_offset, ref_len, indels)`
    tuple returned by `cs_tag_parse()` in python_reference/mcall.py.
    """

    var query_start_offset: Int
    var query_end_offset: Int
    var ref_len: Int
    var indels: List[Indel]

    def __init__(out self):
        self.query_start_offset = 0
        self.query_end_offset = 0
        self.ref_len = 0
        self.indels = List[Indel]()


def _is_cs_op(c: String) -> Bool:
    return c == "+" or c == "-" or c == ":" or c == "*"


def cs_tag_parse(alignment_tag: String) raises -> CsTagResult:
    """Parse a minimap2/vg `cs:Z:` short-form alignment tag.

    Faithful native port of `cs_tag_parse()` in python_reference/mcall.py.
    Splits the tag into `:N` (match run), `*xy` (mismatch), `-seq`
    (deletion), and `+seq` (insertion) tokens, then derives:
      - `query_start_offset` / `query_end_offset`: leading/trailing soft-clip
        implied by a leading `+seq` or trailing `-seq` token,
      - `ref_len`: total reference bases spanned,
      - `indels`: the list of insertion/deletion events (with reference-
        relative start offsets), used by the caller to re-align the query
        read against the reference block base-by-base.
    """
    var res = CsTagResult()

    var tag_parsed = List[String]()
    var element = String("")

    for ch in alignment_tag.codepoints():
        var s = String(ch)
        if _is_cs_op(s):
            if element.byte_length() == 0 and len(tag_parsed) == 0:
                element = s
                continue
            tag_parsed.append(element)
            element = s
            continue
        element += s
    tag_parsed.append(element)

    if len(tag_parsed) > 0 and tag_parsed[0].startswith("+"):
        res.query_start_offset = tag_parsed[0].byte_length() - 1
        _ = tag_parsed.pop(0)
    if len(tag_parsed) > 0 and tag_parsed[len(tag_parsed) - 1].startswith("-"):
        res.query_end_offset = tag_parsed[len(tag_parsed) - 1].byte_length() - 1
        _ = tag_parsed.pop(len(tag_parsed) - 1)

    for item in tag_parsed:
        if item.startswith(":"):
            res.ref_len += Int(item[byte=1 : item.byte_length()])
        elif item.startswith("*"):
            res.ref_len += 1
        elif item.startswith("-"):
            var l = item.byte_length() - 1
            res.indels.append(Indel(res.ref_len, l, "-"))
            res.ref_len += l
        elif item.startswith("+"):
            var l = item.byte_length() - 1
            res.indels.append(Indel(res.ref_len, l, "+"))
            # Matches python_reference/mcall.py: insertions do NOT advance
            # ref_len (they consume query bases only, not reference bases).

    return res^
