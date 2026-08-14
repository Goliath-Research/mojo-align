# engine/alignments.py — faithful port of python_reference/alignments.py
# (methylGrapher 0.2.0). No behavioral patches beyond package-relative imports;
# see MIGRATION_LOG.md.

import os
import sys
import gzip
import hashlib
import resource
import multiprocessing
import time

from . import mcall
from . import utility




tmp_alignment_file_count = 1000


# The OS may have limit of how many file can you open at the same time.
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
assert tmp_alignment_file_count+500 < hard_limit
resource.setrlimit(resource.RLIMIT_NOFILE, (tmp_alignment_file_count+500, hard_limit))












conversion_types = ["C2T", "G2A"]


def _shard_marker_path(shard_path):
    return shard_path + ".done"


def shard_fingerprint(parts):
    """Identity of the map that produced a shard.

    Converted FASTQs are rewritten on every run, so mtime would never match;
    size distinguishes one sample's reads from another's and survives
    regeneration of the same sample.
    """
    fields = []
    for part in parts:
        if part and os.path.exists(part):
            fields.append(f"{part}:{os.path.getsize(part)}")
        else:
            fields.append(f"{part}:-")
    return hashlib.sha256("|".join(fields).encode()).hexdigest()


def mark_shard_complete(shard_path, fingerprint):
    """Record that ``shard_path`` holds the full output of a finished map."""
    with open(_shard_marker_path(shard_path), "w") as fh:
        fh.write(f"{os.path.getsize(shard_path)}\n{fingerprint}\n")


def clear_shard_marker(shard_path):
    try:
        os.remove(_shard_marker_path(shard_path))
    except OSError:
        pass


def shard_is_complete(shard_path, fingerprint):
    """True only for a shard a prior map finished writing for these inputs.

    A killed map leaves a non-empty but truncated GAF, which is
    indistinguishable from a finished one by size alone. The marker records the
    byte count at completion, so a shard that grew or shrank since is remapped,
    and the fingerprint keeps a shard left in a reused work directory from
    standing in for a different sample or index.
    """
    try:
        with open(_shard_marker_path(shard_path)) as fh:
            recorded_size, recorded_fp = fh.read().split()[:2]
        recorded = int(recorded_size)
    except (OSError, ValueError, IndexError):
        return False
    if recorded_fp != fingerprint:
        return False
    try:
        size = os.path.getsize(shard_path)
    except OSError:
        return False
    return size > 0 and size == recorded


def alignment_clenup(work_dir):
    # Converted FASTQ files removal
    for ct in conversion_types:
        for ri in ["R1", "R2"]:
            for fmt in ["fastq", "fastq.gz"]:
                fn = f"{work_dir}/{ct}.{ri}.{fmt}"
                if os.path.exists(fn):
                    os.remove(fn)

    # TMP alignment files removal
    for i in range(tmp_alignment_file_count):
        alignment_out = f"{work_dir}/alignment.{i}.gaf"
        if os.path.exists(alignment_out):
            os.remove(alignment_out)

    return


def alignment(
    work_dir="./",
    index_prefix="",
    output_format="gaf",
    thread=1,
    directional=True,
    compress=True,
    vg_path="vg",
    align_engine=None,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    from .align_backends import resolve_map_command

    index_prefix_ct = index_prefix + ".wl.C2T"
    index_prefix_ga = index_prefix + ".wl.G2A"

    read1_alignment_type = ["C2T"]
    read2_alignment_type = ["G2A"]

    if not directional:
        read1_alignment_type = conversion_types
        read2_alignment_type = conversion_types

    from .align_backends import normalize_align_engine

    mojo_direct = normalize_align_engine(align_engine) in {
        "mojo_giraffe",
        "gpu_giraffe",
        "mojo",
    }
    # MojoGiraffe emits Illumina qnames (no underscore shard index). The classic
    # 1000-way shard router silently drops every line — write alignment.gaf directly.
    alignment_file_path = []
    alignment_outs = {}
    if not mojo_direct:
        for i in range(tmp_alignment_file_count):
            alignment_out = f"{work_dir}/alignment.{i}.{output_format}"
            alignment_file_path.append(alignment_out)
            fh = open(alignment_out, "w")
            alignment_outs[i] = fh

    shard_lock = threading.Lock()
    final_gaf = os.path.join(work_dir, "alignment.gaf")
    if mojo_direct and os.path.exists(final_gaf):
        os.remove(final_gaf)

    def _run_one_map(ref_type, read_type1, read_type2):
        print(f"Aligning R1({read_type1}) & R2({read_type2}) on reference({ref_type})")

        fq1 = f"{work_dir}/{read_type1}.R1.fastq"
        fq2 = f"{work_dir}/{read_type2}.R2.fastq"

        if compress:
            fq1 += ".gz"
            fq2 += ".gz"

        # Cheap PE length preflight (sidecar .n_reads from convert) before GPU map.
        if os.path.exists(fq1) and os.path.exists(fq2):
            n1 = utility._converted_fastq_n_reads(fq1)
            n2 = utility._converted_fastq_n_reads(fq2)
            if n1 is not None and n2 is not None and n1 != n2:
                raise RuntimeError(
                    f"PE FASTQ pair-count mismatch before map: {fq1} n={n1} vs {fq2} n={n2}"
                )

        job_index_prefix = index_prefix_ga if ref_type == "G2A" else index_prefix_ct

        giraffe_input = f"-f {fq1}"
        if os.path.exists(fq2):
            giraffe_input += f" -f {fq2}"

        alignment_log = (
            f"{work_dir}/alignment.Ref_{ref_type}.R1_{read_type1}.R2_{read_type2}.log"
        )
        mojo_out_gaf = os.path.join(
            work_dir,
            f"alignment.mojo.Ref_{ref_type}.R1_{read_type1}.R2_{read_type2}.gaf",
        )

        dist_fp = f"{job_index_prefix}.dist"
        gbz_fp = f"{job_index_prefix}.giraffe.gbz"

        min1_fp = f"{job_index_prefix}.min"
        min2_fp = f"{job_index_prefix}.shortread.withzip.min"
        zipcode_fp = f"{job_index_prefix}.shortread.zipcodes"

        assert os.path.exists(dist_fp)
        assert os.path.exists(gbz_fp)

        assert os.path.exists(min1_fp) or os.path.exists(min2_fp)
        if os.path.exists(min2_fp):
            assert os.path.exists(zipcode_fp)

        index_params = f"-Z {gbz_fp} -d {dist_fp}"
        if os.path.exists(min1_fp):
            index_params += f" -m {min1_fp}"
        else:
            index_params += f" -m {min2_fp} -z {zipcode_fp}"

        engine_used, cmd = resolve_map_command(
            align_engine=align_engine,
            vg_path=vg_path,
            thread=thread,
            output_format=output_format,
            index_params=index_params,
            giraffe_input=giraffe_input,
            index_prefix=job_index_prefix,
            out_gaf=mojo_out_gaf if mojo_direct else None,
        )
        print(f"Align map backend: {engine_used} ref={ref_type}")
        # Drop leftover log (often root-owned from prior --user 0:0 runs) before rewrite.
        try:
            if os.path.exists(alignment_log):
                os.remove(alignment_log)
        except OSError as exc:
            print(f"WARNING: could not remove stale align log {alignment_log}: {exc}")
        with open(alignment_log, "w") as alignment_log_fh:
            alignment_log_fh.write("Command used: \n")
            alignment_log_fh.write(cmd + "\n\n")
            alignment_log_fh.write(f"Backend: {engine_used}\n")

        cmd_run = "\n".join(
            ln for ln in cmd.splitlines() if ln.strip() and not ln.strip().startswith("#")
        )
        if "MojoGiraffe" in cmd_run or cmd_run.lstrip().startswith("set "):
            import shlex

            cmd_run = "bash -lc " + shlex.quote(cmd_run)
        se = utility.SystemExecute()

        if mojo_direct or "MojoGiraffe" in cmd_run:
            shard_id = shard_fingerprint([fq1, fq2, gbz_fp, dist_fp, engine_used])

            # Resume: keep a completed Mojo shard (e.g. C2T done, G2A aborted on
            # orchestration bug) instead of remapping for another ~hour.
            if (
                shard_is_complete(mojo_out_gaf, shard_id)
                and os.environ.get("METHYLGRAPHER_ALIGN_FORCE_REMAP", "").strip()
                not in {"1", "true", "yes"}
            ):
                print(
                    f"Reusing Mojo GAF shard {mojo_out_gaf} "
                    f"({os.path.getsize(mojo_out_gaf)} bytes); skip remap",
                    flush=True,
                )
                with shard_lock:
                    with open(mojo_out_gaf, "r", encoding="utf-8") as src, open(
                        final_gaf, "a", encoding="utf-8"
                    ) as dst:
                        for line in src:
                            dst.write(line if line.endswith("\n") else line + "\n")
                print(
                    f"Mojo GAF appended -> {final_gaf} "
                    f"(+{os.path.getsize(mojo_out_gaf)} bytes from {mojo_out_gaf})",
                    flush=True,
                )
                return ref_type, engine_used

            if os.path.isfile(mojo_out_gaf) and os.path.getsize(mojo_out_gaf) > 0:
                print(
                    f"Mojo GAF shard {mojo_out_gaf} is not a reusable completed "
                    f"map for these inputs; remapping ref={ref_type}",
                    flush=True,
                )

            # A marker from an earlier attempt must not outlive the shard it
            # describes, in case this map is killed partway through.
            clear_shard_marker(mojo_out_gaf)

            # File-backed GAF; Mojo logs on stderr. Do not use underscore shard router.
            fout, flog = se.execute(cmd_run, stdout=None, stderr=alignment_log)
            # Drain stdout (usually empty when out_gaf is a file).
            if fout is not None:
                for _ in fout:
                    pass
            codes = se.wait()
            if any(rc != 0 for rc in codes):
                raise RuntimeError(
                    f"Align map command failed (exit {codes}) backend={engine_used}; "
                    f"see {alignment_log}"
                )
            if not os.path.isfile(mojo_out_gaf) or os.path.getsize(mojo_out_gaf) == 0:
                raise RuntimeError(
                    f"MojoGiraffe produced empty GAF {mojo_out_gaf}; see {alignment_log}"
                )
            mark_shard_complete(mojo_out_gaf, shard_id)
            with shard_lock:
                with open(mojo_out_gaf, "r", encoding="utf-8") as src, open(
                    final_gaf, "a", encoding="utf-8"
                ) as dst:
                    for line in src:
                        dst.write(line if line.endswith("\n") else line + "\n")
            # ASCII-only: docker/worker capture often uses LANG=C (ascii stdout).
            print(
                f"Mojo GAF appended -> {final_gaf} "
                f"(+{os.path.getsize(mojo_out_gaf)} bytes from {mojo_out_gaf})"
            )
            return ref_type, engine_used

        fout, flog = se.execute(cmd_run, stdout=None, stderr=alignment_log)
        for line in fout:
            line = line.decode("utf-8")
            stripped = line.strip()
            if not stripped:
                continue
            l = stripped.split("\t")
            if len(l) < 12:
                continue

            asterisk = False
            for i in [2, 3, 6, 7, 8, 9, 10, 11]:
                if l[i] == "*":
                    asterisk = True
                    break
            if asterisk:
                continue

            name_parts = l[0].split("_")
            if len(name_parts) < 3:
                continue
            try:
                ind = int(name_parts[2])
            except ValueError:
                continue
            with shard_lock:
                if ind not in alignment_outs:
                    continue
                alignment_out_fh = alignment_outs[ind]
                alignment_out_fh.write(line if line.endswith("\n") else line + "\n")

        codes = se.wait()
        if any(rc != 0 for rc in codes):
            raise RuntimeError(
                f"Align map command failed (exit {codes}) backend={engine_used}; "
                f"see {alignment_log}"
            )
        return ref_type, engine_used

    jobs = []
    for ref_type in conversion_types:
        for read_type1 in read1_alignment_type:
            for read_type2 in read2_alignment_type:
                if read_type1 == read_type2:
                    continue
                jobs.append((ref_type, read_type1, read_type2))

    # Default serialize C2T/G2A when GPU/Mojo DeviceContext is in use — two
    # concurrent CUDA contexts on one GH200 cause CUDA_ERROR_ILLEGAL_ADDRESS.
    # Opt in with METHYLGRAPHER_DUAL_GRAPH_PARALLEL=1 only after GPU isolation.
    parallel_raw = os.environ.get("METHYLGRAPHER_DUAL_GRAPH_PARALLEL", "").strip().lower()
    if parallel_raw == "":
        device = os.environ.get("METHYLGRAPHER_GIRAFFE_DEVICE", "").strip().lower()
        # mojo_direct covers the engine passed as an argument as well as
        # METHYLGRAPHER_ALIGN_ENGINE; reading the env alone would let a
        # CLI-selected GPU engine run both graphs at once.
        gpuish = device in {"nvidia", "amd", "cuda", "hip", "rocm"} or mojo_direct
        parallel = not gpuish
    else:
        parallel = parallel_raw not in {"0", "false", "no", "off"}
    workers = 2 if parallel and len(jobs) > 1 else 1
    print(f"Align dual-graph jobs={len(jobs)} parallel_workers={workers}")

    def _reclaim_hbm_between_graphs() -> None:
        """Wait for CUDA to free HBM after one MojoGiraffe process exits.

        Without this, the second dual-graph job (G2A) often sees only a few GiB
        free and trips the capacity preflight / DeviceContext OOM.
        """
        if workers != 1:
            return
        # Deliberately env-only: inferring a GPU from the engine argument would
        # make every serialized run poll wait_for_hbm_free for its full timeout
        # on hosts whose HBM is busy or absent.
        device = os.environ.get("METHYLGRAPHER_GIRAFFE_DEVICE", "").strip().lower()
        engine = os.environ.get("METHYLGRAPHER_ALIGN_ENGINE", "").strip().lower()
        if device in {"cuda", ""}:
            device = "nvidia" if device == "cuda" or engine in {
                "gpu_giraffe",
                "mojo_giraffe",
                "mojo",
            } else device
        if device in {"hip", "rocm"}:
            device = "amd"
        if device not in {"nvidia", "amd"}:
            return
        try:
            from engine import gpu_mem
        except ImportError:
            return
        timeout = float(os.environ.get("METHYLGRAPHER_GRAPH_HANDOFF_TIMEOUT_S", "180"))
        # Budget is a share of whatever HBM this device actually has — the same
        # operator pin the capacity preflight uses. The absolute GiB env stays as
        # an escape hatch; there is no code fallback, because a fixed GiB target
        # silently becomes unreachable on a smaller (or busier) GPU.
        absolute = os.environ.get("METHYLGRAPHER_GRAPH_HANDOFF_FREE_GIB", "").strip()
        if absolute:
            min_free = float(absolute)
            budget = f"{min_free:.1f} GiB (absolute pin)"
        else:
            fraction = gpu_mem.hbm_fraction_from_env()
            if fraction is None:
                print(
                    "Align dual-graph HBM handoff: no budget pinned "
                    "(METHYLGRAPHER_GPU_HBM_FRACTION / "
                    "METHYLGRAPHER_GRAPH_HANDOFF_FREE_GIB); deferring to the next "
                    "graph's capacity preflight",
                    flush=True,
                )
                return
            min_free = gpu_mem.fraction_min_free_gib(fraction, device=device)
            budget = f"{min_free:.1f} GiB ({fraction:.0%} of total)"
        print(
            f"Align dual-graph HBM handoff: waiting for ≥{budget} free "
            f"(device={device})",
            flush=True,
        )
        gpu_mem.wait_for_hbm_free(min_free, device=device, timeout_s=timeout)

    if workers == 1:
        # Host-only: start graph-2 pack resolve while graph-1 maps on the GPU.
        # Never start a second DeviceContext / MojoGiraffe here.
        next_pack_fut = None
        if len(jobs) > 1:
            try:
                from engine.quartet_map import ensure_pack_for_gbz as _ensure_pack
            except ImportError:
                try:
                    from quartet_map import ensure_pack_for_gbz as _ensure_pack  # type: ignore
                except ImportError:
                    _ensure_pack = None  # type: ignore
            if _ensure_pack is not None:
                def _gbz_for_job(job) -> str:
                    ref_type = job[0]
                    prefix = index_prefix_ga if ref_type == "G2A" else index_prefix_ct
                    return f"{prefix}.giraffe.gbz"

                # Ensure graph-1 pack is ready, then prefetch graph-2 in background.
                try:
                    _ = _ensure_pack(_gbz_for_job(jobs[0]))
                except Exception as exc:
                    print(f"WARNING: pack ensure job0: {exc}", flush=True)
                pref_pool = ThreadPoolExecutor(max_workers=1)
                next_pack_fut = pref_pool.submit(_ensure_pack, _gbz_for_job(jobs[1]))
                print(
                    "Align dual-graph: prefetching pack for job1 while job0 maps",
                    flush=True,
                )
            else:
                pref_pool = None
        else:
            pref_pool = None
        for i, job in enumerate(jobs):
            if i > 0:
                if next_pack_fut is not None:
                    try:
                        _ = next_pack_fut.result()
                        print("Align dual-graph: job1 pack prefetch ready", flush=True)
                    except Exception as exc:
                        print(f"WARNING: pack prefetch job1: {exc}", flush=True)
                    next_pack_fut = None
                    if pref_pool is not None:
                        pref_pool.shutdown(wait=False)
                        pref_pool = None
                _reclaim_hbm_between_graphs()
            _run_one_map(*job)
        if pref_pool is not None:
            pref_pool.shutdown(wait=False)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_run_one_map, *job) for job in jobs]
            for fut in as_completed(futs):
                _ = fut.result()

    for fh in alignment_outs.values():
        fh.close()

    if mojo_direct:
        _finalize_mojo_gaf_named_coordinates(final_gaf)

    return


def _finalize_mojo_gaf_named_coordinates(final_gaf: str) -> None:
    """Rewrite Mojo GBZ node ids → GFA named-coordinates before MethylCall.

    Align packs use ``vg convert --no-translation``; MethylCall + cpg.tsv use
    ``*.wl.gfa`` segment ids. Classic giraffe emits ``--named-coordinates``;
    Mojo must finalize the concatenated GAF the same way.
    """
    if not final_gaf or not os.path.isfile(final_gaf) or os.path.getsize(final_gaf) <= 0:
        return
    try:
        from . import named_coords
    except ImportError:
        from engine import named_coords  # type: ignore
    named_coords.ensure_translated_inplace(final_gaf)


def tmp_gaf_processing(tmp_gaf_fp):
    reads = {}
    result_gaf_str = ''
    with open(tmp_gaf_fp) as f:
        for i, l in enumerate(f):
            l = l.strip().split('\t')
            newl = l[:]

            query_name_complex = l[0].split('_')
            query_name = query_name_complex[0]
            read_conversion = query_name_complex[1][0] + query_name_complex[1][2]
            original_seq = query_name_complex[3]

            newl[0] = query_name

            pop_i = -1
            r1 = False
            r2 = False
            for j, e in enumerate(newl):
                if e.startswith("fn:Z:"):
                    r1 = True
                    pop_i = j
                if e.startswith("fp:Z:"):
                    r2 = True
                    pop_i = j

            assert not (r1 and r2)
            newl.pop(pop_i)

            ri = 1 if r1 else 2
            ri_tag = f'ri:i:{ri}'
            newl.append(ri_tag)

            newl.append(f'os:Z:{original_seq}')

            newl.append(f'rc:Z:{read_conversion}')

            if query_name not in reads:
                reads[query_name] = [[], []]

            reads[query_name][ri - 1].append(newl)

    counter = [0, 0, 0, 0, 0]
    for query_name, read_pair_alignments in reads.items():
        # print(f'Processing {query_name}')

        for read_alignments in read_pair_alignments:
            counter[0] += 1

            if len(read_alignments) == 0:
                counter[1] += 1
                continue

            elif len(read_alignments) == 1:
                counter[2] += 1
                read_alignment = read_alignments[0]
                result_gaf_str += '\t'.join(read_alignment) + '\n'
                continue

            else:
                # Determine best alignment
                best_alignments = []
                best_score = -1

                for read_alignment in read_alignments:
                    mapq = int(read_alignment[11])

                    if mapq > best_score:
                        best_score = mapq

                for read_alignment in read_alignments:
                    mapq = int(read_alignment[11])
                    if mapq == best_score:
                        best_alignments.append(read_alignment)

                if len(best_alignments) == 1:
                    counter[3] += 1
                    result_gaf_str += '\t'.join(best_alignments[0]) + '\n'
                    continue

                # Determine multimapping from now on
                gaf_line_set = set()
                for read_alignment in best_alignments:
                    gaf_line = '\t'.join(read_alignment) + '\n'
                    gaf_line_set.add(gaf_line)

                # Are all the alignments the same? If so, just output one, and it is not multimapping
                if len(gaf_line_set) == 1:
                    counter[3] += 1
                    result_gaf_str += '\t'.join(best_alignments[0]) + '\n'
                    continue

                # distinguish by alignment score.
                best_score = -1
                best_alignments_by_as = []
                for read_alignment in best_alignments:
                    for e in read_alignment:
                        if e.startswith('AS:i:'):
                            score = int(e.split(':')[-1])
                            if score > best_score:
                                best_score = score

                for read_alignment in best_alignments:
                    for e in read_alignment:
                        if e.startswith('AS:i:'):
                            score = int(e.split(':')[-1])
                            if score == best_score:
                                best_alignments_by_as.append(read_alignment)

                if len(best_alignments_by_as) == 1:
                    counter[3] += 1
                    result_gaf_str += '\t'.join(best_alignments_by_as[0]) + '\n'
                    continue

                # Do alignment share any common segments? If they do, it is multi-path alignment.
                # And I do not consider them as multimapping
                shared_segments = set()
                init = True
                for read_alignment in best_alignments_by_as:
                    path_str = read_alignment[5]
                    path = mcall.alignment_path_parse(path_str)
                    # print(path)

                    if init:
                        shared_segments = set(path[0])
                        init = False

                    shared_segments = shared_segments.intersection(set(path[0]))

                if len(shared_segments) > 0:
                    result_gaf_str += '\t'.join(best_alignments_by_as[0]) + '\n'
                    counter[3] += 1
                    continue

                # Hmm, still multimapping. Let's just output one of them.
                result_gaf_str += '\t'.join(best_alignments_by_as[0]) + "\tmp:i:1" + '\n'
                counter[4] += 1

    # counter2 = counter[:]
    # for ic in range(len(counter2)):
    #    counter2[ic] = counter2[ic] / counter[0] * 100

    return result_gaf_str


def tmp_gaf_processing_worker(pid, input_queue, result_queue):
    while True:
        tmp_gaf_fp = input_queue.get()
        if tmp_gaf_fp is None:
            result_queue.put(None)
            break

        result_gaf_str = tmp_gaf_processing(tmp_gaf_fp)
        result_queue.put(result_gaf_str)
    return

def merger_gaf_writer_worker(pid, worker_num, result_queue, output_fp):
    counter = 0
    with open(output_fp, 'w') as fh:
        while True:
            result_gaf_str = result_queue.get()
            if result_gaf_str is None:
                counter += 1
                if counter == worker_num:
                    break
                continue
            fh.write(result_gaf_str)
    return


def alignment_merge_main(working_dir, worker_num=20):
    output_gaf_fp = os.path.join(working_dir, 'alignment.gaf')

    input_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    pool = []
    for i in range(worker_num):
        p = multiprocessing.Process(target=tmp_gaf_processing_worker, args=(i, input_queue, result_queue))
        p.start()
        pool.append(p)

    writer_worker = multiprocessing.Process(target=merger_gaf_writer_worker, args=(0, worker_num, result_queue, output_gaf_fp))
    writer_worker.start()

    for i in range(tmp_alignment_file_count):
        input_gaf_fp = os.path.join(working_dir, f'alignment.{i}.gaf')
        input_queue.put(input_gaf_fp)

    for i in range(worker_num):
        input_queue.put(None)

    for p in pool:
        p.join()

    writer_worker.join()

    return


def alignment_main(
    fq1,
    fq2,
    work_dir,
    index_prefix,
    compress=True,
    thread=1,
    directional=True,
    vg_path="vg",
    align_engine=None,
):
    utility.fastq_converter(fq1, fq2, work_dir,
                            compress=compress,
                            thread=thread,
                            directional=directional,
                            split_num=tmp_alignment_file_count)

    from .align_backends import normalize_align_engine

    alignment(work_dir=work_dir,
              index_prefix=index_prefix,
              output_format="gaf",
              thread=thread,
              directional=directional,
              compress=compress,
              vg_path=vg_path,
              align_engine=align_engine)

    # Mojo path already wrote work_dir/alignment.gaf; classic merge expects
    # underscore-encoded shard qnames and would wipe/empty the Mojo GAF.
    if normalize_align_engine(align_engine) not in {"mojo_giraffe", "gpu_giraffe", "mojo"}:
        alignment_merge_main(work_dir, worker_num=thread)
        time.sleep(5)

    alignment_clenup(work_dir)

    return


if __name__ == '__main__':
    working_dir = sys.argv[1]
    alignment_merge_main(working_dir, worker_num=20)
    sys.exit(0)


































































































