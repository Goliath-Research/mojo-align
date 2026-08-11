# Compatibility shim — device selection lives in gpu_device (gpu-common).

from std.collections import List

from gpu_device import (
    extract_kmers_batch as _extract_kmers_batch,
    extract_kmers_batch_cpu as _extract_kmers_batch_cpu,
    require_device_or_raise as _require_device_or_raise,
    select_device as _select_device,
)


def select_device(requested: String) raises -> String:
    return _select_device(requested)


def require_device_or_raise(device: String) raises -> String:
    return _require_device_or_raise(device)


def extract_kmers_batch_cpu(seqs: List[String], k: Int) raises -> List[List[String]]:
    return _extract_kmers_batch_cpu(seqs, k)


def extract_kmers_batch(
    device: String, seqs: List[String], k: Int
) raises -> List[List[String]]:
    return _extract_kmers_batch(device, seqs, k)
