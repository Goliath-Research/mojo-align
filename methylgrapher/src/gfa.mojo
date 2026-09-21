# src/gfa.mojo
# Native Mojo segment-sequence lookup for MethylCall.
#
# Ports the MethylCall-critical surface of
# `engine.gfa.GraphicalFragmentAssemblyMemory`: load `S` lines into a
# `Dict[String, String]` so the hot loop never keeps the ~43 GiB graph only
# on the Python side. Links / walks / SNV trimming stay in `engine/gfa.py`.

from std.collections import Dict, List

from utility import open_text_read


struct GraphicalFragmentAssemblyMemory(Copyable, Movable):
    """In-memory GFA segment sequences keyed by segment ID.

    Faithful subset of `GraphicalFragmentAssemblyMemory` in engine/gfa.py
    for MethylCall: parse `S` records, look up one or many sequences.
    """

    var segments: Dict[String, String]
    var original_gfa_path: String

    def __init__(out self):
        self.segments = Dict[String, String]()
        self.original_gfa_path = String("")

    def clear(mut self):
        self.segments = Dict[String, String]()
        self.original_gfa_path = String("")

    def parse(mut self, gfa_file: String) raises:
        """Load segment sequences from a (optionally gzipped) GFA file.

        Only `S` lines are retained — matching MethylCall's use of
        `get_sequences_by_segment_ID`. Header / link / walk lines are skipped.
        """
        self.clear()
        self.original_gfa_path = gfa_file

        var fh = open_text_read(gfa_file)
        while True:
            var line_obj = fh.readline()
            var line = String(line_obj)
            if line.byte_length() == 0:
                break
            if not line.startswith("S"):
                continue
            while line.byte_length() > 0:
                var last = String(
                    line[byte = line.byte_length() - 1 : line.byte_length()]
                )
                if last == "\n" or last == "\r":
                    var trimmed = String(line[byte = 0 : line.byte_length() - 1])
                    line = trimmed
                else:
                    break
            var parts = line.split("\t")
            if len(parts) < 3:
                continue
            if String(parts[0]) != "S":
                continue
            var seg_id = String(parts[1])
            var seq = String(parts[2])
            self.segments[seg_id] = seq
        fh.close()

    def get_sequence_by_segment_ID(self, segment_ID: String) raises -> String:
        if segment_ID not in self.segments:
            raise Error("GFA segment not found: " + segment_ID)
        return self.segments[segment_ID].copy()

    def get_sequences_by_segment_ID(
        self, segment_IDs: List[String]
    ) raises -> Dict[String, String]:
        var res = Dict[String, String]()
        for sID in segment_IDs:
            res[sID] = self.get_sequence_by_segment_ID(sID)
        return res^

    def segment_count(self) -> Int:
        return len(self.segments)
