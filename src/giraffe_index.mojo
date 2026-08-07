# Graph + in-memory k-mer index for Mojo Giraffe (GFA-native development path).

from std.collections import Dict, List

from utility import open_text_read


struct GraphLink(Copyable, Movable):
    var frm: String
    var fro: String
    var too: String
    var to_o: String

    def __init__(out self, frm: String, fro: String, too: String, to_o: String):
        self.frm = frm
        self.fro = fro
        self.too = too
        self.to_o = to_o


struct GraphIndex(Copyable, Movable):
    var segments: Dict[String, String]
    var links: List[GraphLink]
    # Flattened hit table: "kmer\tseg:offset" rows for fixture-scale indexes.
    var hit_table: List[String]
    var k: Int
    var w: Int

    def __init__(out self, k: Int = 5, w: Int = 5):
        self.segments = Dict[String, String]()
        self.links = List[GraphLink]()
        self.hit_table = List[String]()
        self.k = k
        self.w = w

    def load_gfa(mut self, gfa_path: String) raises:
        self.segments = Dict[String, String]()
        self.links = List[GraphLink]()
        var fh = open_text_read(gfa_path)
        while True:
            var line_obj = fh.readline()
            var line = String(line_obj)
            if line.byte_length() == 0:
                break
            while line.byte_length() > 0:
                var last = String(line[byte = line.byte_length() - 1 : line.byte_length()])
                if last == "\n" or last == "\r":
                    line = String(line[byte = 0 : line.byte_length() - 1])
                else:
                    break
            if line.byte_length() == 0:
                continue
            var parts = line.split("\t")
            if len(parts) < 1:
                continue
            var kind = String(parts[0])
            if kind == "S" and len(parts) >= 3:
                self.segments[String(parts[1])] = String(parts[2])
            elif kind == "L" and len(parts) >= 5:
                self.links.append(
                    GraphLink(
                        String(parts[1]),
                        String(parts[2]),
                        String(parts[3]),
                        String(parts[4]),
                    )
                )
        fh.close()

    def build_minimizer_index(mut self) raises:
        self.hit_table = List[String]()
        var seg_ids = List[String]()
        for seg_id in self.segments:
            seg_ids.append(seg_id)
        for seg_id in seg_ids:
            var seq = self.segments[seg_id].copy()
            var n = seq.byte_length()
            if n < self.k:
                continue
            var i = 0
            while i <= n - self.k:
                var mer = String(seq[byte = i : i + self.k])
                self.hit_table.append(mer + "\t" + seg_id + ":" + String(i))
                i += 1

    def lookup_kmer(self, mer: String) raises -> List[String]:
        var hits = List[String]()
        for row in self.hit_table:
            var parts = row.split("\t")
            if len(parts) >= 2 and String(parts[0]) == mer:
                hits.append(String(parts[1]))
        return hits^

    def segment_count(self) -> Int:
        var n = 0
        for _k in self.segments:
            n += 1
        return n

    def segment_ids(self) raises -> List[String]:
        var ids = List[String]()
        for seg_id in self.segments:
            ids.append(seg_id)
        return ids^
