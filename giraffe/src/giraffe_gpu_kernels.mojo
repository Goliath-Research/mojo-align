# Compatibility shim — portable kernels live in gpu_kernels (gpu-common).

from std.collections import List

from gpu_kernels import (
    kernel_target_label as _kernel_target_label,
    last_gpu_backend as _last_gpu_backend,
    probe_device_context as _probe_device_context,
    seed_kmers_on_device as _seed_kmers_on_device,
    seed_kmers_portable as _seed_kmers_portable,
)


def kernel_target_label(device: String) raises -> String:
    return _kernel_target_label(device)


def probe_device_context(device: String) raises -> String:
    return _probe_device_context(device)


def seed_kmers_portable(seqs: List[String], k: Int) raises -> List[List[String]]:
    return _seed_kmers_portable(seqs, k)


def last_gpu_backend() raises -> String:
    return _last_gpu_backend()


def seed_kmers_on_device(
    device: String, seqs: List[String], k: Int
) raises -> List[List[String]]:
    return _seed_kmers_on_device(device, seqs, k)
