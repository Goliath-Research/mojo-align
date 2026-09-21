# Address slot for Mojo 1.1 `parallelize`.
#
# `max.algorithm.parallelize` accepts only a non-capturing `def(Int) -> None`.
# Module globals are rejected, so the caller publishes a stack address through
# the process environment for the duration of the call. `parallelize` is
# synchronous, and the pointed-to value stays on the caller stack until the
# workers return.

from std.os import getenv, setenv, unsetenv


def set_parallel_slot(name: String, addr: Int) raises:
    _ = setenv(name, String(addr), True)


def clear_parallel_slot(name: String) raises:
    _ = unsetenv(name)


def parallel_slot_addr(name: String) -> Int:
    try:
        return Int(getenv(name, "0"))
    except:
        return 0
