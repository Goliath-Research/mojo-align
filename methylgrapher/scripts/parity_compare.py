#!/usr/bin/env python
"""scripts/parity_compare.py

Compare the outputs of two methylGrapher-mojo `MethylCall`/`MergeCpG` runs
(e.g. Python engine vs. Mojo dispatch, or before/after porting a pipeline
stage natively to Mojo) for exact parity.

Compares:
  - graph.methyl   (segID, pos, strand, context, unmet, met, cov, ml)
  - graph.cpg.tsv  (cpg_id, met, cov [, segID1, pos1, segID2, pos2 if
                    generated with full_position=True])

Row order is not significant (methylCall's multiprocess workers may emit
rows in different orders across runs/engines); rows are compared as sets
keyed by their identifying columns.

Usage:
    python scripts/parity_compare.py --a-work-dir DIR_A --b-work-dir DIR_B
    python scripts/parity_compare.py --methyl-a A/graph.methyl --methyl-b B/graph.methyl --skip-cpg

Exit code is 0 if all compared files match exactly, 1 otherwise.
"""

import argparse
import os
import sys


def read_graph_methyl(path):
    """Parse a graph.methyl file into {(segID, pos, strand, context): (unmet, met, cov, ml)}."""
    data = {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) != 8:
                raise ValueError(f"{path}:{lineno}: expected 8 columns in graph.methyl, got {len(cols)}: {line!r}")
            seg_id, pos, strand, context, unmet, met, cov, ml = cols
            key = (seg_id, pos, strand, context)
            if key in data:
                raise ValueError(f"{path}:{lineno}: duplicate row for {key}")
            data[key] = (int(unmet), int(met), int(cov), float(ml))
    return data


def read_graph_cpg(path):
    """Parse a graph.cpg.tsv file into {cpg_id: (met, cov)}.

    Only the cpg_id (first column) and the trailing met/cov columns are
    used, so this works for both `full_position=False` (cpg_id, met, cov)
    and `full_position=True` (cpg_id, segID1, pos1, segID2, pos2, met, cov)
    output shapes.
    """
    data = {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                raise ValueError(f"{path}:{lineno}: expected >= 3 columns in graph.cpg.tsv, got {len(cols)}: {line!r}")
            cpg_id = cols[0]
            met, cov = cols[-2], cols[-1]
            if cpg_id in data:
                raise ValueError(f"{path}:{lineno}: duplicate row for cpg_id {cpg_id}")
            data[cpg_id] = (int(met), int(cov))
    return data


def compare(a, b, label, max_report=20):
    a_keys = set(a)
    b_keys = set(b)
    only_a = sorted(a_keys - b_keys, key=str)
    only_b = sorted(b_keys - a_keys, key=str)
    mismatched = [(k, a[k], b[k]) for k in sorted(a_keys & b_keys, key=str) if a[k] != b[k]]

    ok = not only_a and not only_b and not mismatched

    print(f"[{label}] A={len(a)} rows, B={len(b)} rows")

    def report(title, items, fmt):
        print(f"  {title} ({len(items)}):")
        for entry in items[:max_report]:
            print(f"    {fmt(entry)}")
        if len(items) > max_report:
            print(f"    ... and {len(items) - max_report} more")

    if only_a:
        report("only in A", only_a, lambda k: f"{k}: {a[k]}")
    if only_b:
        report("only in B", only_b, lambda k: f"{k}: {b[k]}")
    if mismatched:
        report("value mismatches", mismatched, lambda e: f"{e[0]}: A={e[1]} B={e[2]}")

    print("  OK: identical" if ok else "  MISMATCH")
    return ok


def _resolve(explicit, work_dir, filename):
    if explicit:
        return explicit
    if work_dir:
        return os.path.join(work_dir, filename)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a-work-dir", help="work_dir of run A (looks for graph.methyl / graph.cpg.tsv)")
    parser.add_argument("--b-work-dir", help="work_dir of run B")
    parser.add_argument("--methyl-a", help="explicit path to run A's graph.methyl (overrides --a-work-dir)")
    parser.add_argument("--methyl-b", help="explicit path to run B's graph.methyl (overrides --b-work-dir)")
    parser.add_argument("--cpg-a", help="explicit path to run A's graph.cpg.tsv (overrides --a-work-dir)")
    parser.add_argument("--cpg-b", help="explicit path to run B's graph.cpg.tsv (overrides --b-work-dir)")
    parser.add_argument("--skip-methyl", action="store_true", help="don't compare graph.methyl")
    parser.add_argument("--skip-cpg", action="store_true", help="don't compare graph.cpg.tsv")
    args = parser.parse_args(argv)

    methyl_a = _resolve(args.methyl_a, args.a_work_dir, "graph.methyl")
    methyl_b = _resolve(args.methyl_b, args.b_work_dir, "graph.methyl")
    cpg_a = _resolve(args.cpg_a, args.a_work_dir, "graph.cpg.tsv")
    cpg_b = _resolve(args.cpg_b, args.b_work_dir, "graph.cpg.tsv")

    all_ok = True
    compared_anything = False

    if not args.skip_methyl:
        if not methyl_a or not methyl_b:
            parser.error("need --a-work-dir/--b-work-dir or --methyl-a/--methyl-b to compare graph.methyl (or pass --skip-methyl)")
        all_ok = compare(read_graph_methyl(methyl_a), read_graph_methyl(methyl_b), "graph.methyl") and all_ok
        compared_anything = True

    if not args.skip_cpg:
        if not cpg_a or not cpg_b:
            parser.error("need --a-work-dir/--b-work-dir or --cpg-a/--cpg-b to compare graph.cpg.tsv (or pass --skip-cpg)")
        all_ok = compare(read_graph_cpg(cpg_a), read_graph_cpg(cpg_b), "graph.cpg.tsv") and all_ok
        compared_anything = True

    if not compared_anything:
        parser.error("nothing to compare: both --skip-methyl and --skip-cpg given")

    print()
    print("PARITY OK" if all_ok else "PARITY MISMATCH")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
