# Shared alignment hit type for Giraffe GFA/GBZ paths.


struct AlignmentHit(Copyable, Movable):
    var query_name: String
    var path: String
    var qlen: Int
    var mapq: Int
    var cs_tag: String
    var extra_tags: String

    def __init__(
        out self,
        query_name: String,
        path: String,
        qlen: Int,
        mapq: Int,
        cs_tag: String,
        extra_tags: String = "",
    ):
        self.query_name = query_name
        self.path = path
        self.qlen = qlen
        self.mapq = mapq
        self.cs_tag = cs_tag
        self.extra_tags = extra_tags
