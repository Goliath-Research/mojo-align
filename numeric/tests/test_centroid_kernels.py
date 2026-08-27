"""Host-reference parity for centroid stream kernels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from centroid_kernels import (  # noqa: E402
    bin_histogram_add,
    digitize_bins,
    merge_centroid_positions,
    scatter_add_f32,
    scatter_add_u32,
)


def test_scatter_add_u32_duplicates():
    acc = np.zeros(4, dtype=np.uint32)
    scatter_add_u32(acc, np.array([1, 1, 2], dtype=np.intp), np.array([3, 4, 5], dtype=np.uint32))
    np.testing.assert_array_equal(acc, [0, 7, 5, 0])


def test_scatter_add_f32():
    acc = np.zeros(3, dtype=np.float32)
    scatter_add_f32(acc, np.array([0, 2], dtype=np.intp), np.array([0.5, 1.5], dtype=np.float32))
    np.testing.assert_allclose(acc, [0.5, 0.0, 1.5])


def test_bin_histogram_add():
    counts = np.zeros((2, 4), dtype=np.uint32)
    bin_histogram_add(
        counts,
        np.array([0, 0, 1], dtype=np.intp),
        np.array([1, 1, 3], dtype=np.intp),
    )
    assert counts[0, 1] == 2
    assert counts[1, 3] == 1


def test_digitize_bins_matches_numpy():
    edges = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    mean = np.array([0.0, 0.3, 0.99], dtype=np.float32)
    got = digitize_bins(mean, edges)
    expect = np.clip(np.digitize(mean.astype(np.float64), edges[1:-1]), 0, 3)
    np.testing.assert_array_equal(got, expect)


def test_merge_existing_detects_new_positions():
    existing = np.array([10, 20, 40, 0, 0], dtype=np.uint32)
    pos, is_new, n_new = merge_centroid_positions(
        existing, 3, np.array([10, 30, 40], dtype=np.uint32)
    )
    np.testing.assert_array_equal(is_new, [False, True, False])
    assert n_new == 1
    np.testing.assert_array_equal(pos, [10, 30, 40])


def test_merge_first_sample_marks_nonzero_new():
    existing = np.zeros(8, dtype=np.uint32)
    pos, is_new, n_new = merge_centroid_positions(
        existing, 0, np.array([10, 20, 0], dtype=np.uint32)
    )
    np.testing.assert_array_equal(pos, [10, 20, 0])
    assert int(is_new[0]) == 1
    assert int(is_new[1]) == 1
    assert n_new >= 2
