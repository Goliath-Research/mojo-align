"""MOJO_ALIGN_* dual-read helper."""

from __future__ import annotations


def test_getenv_prefers_new_name(monkeypatch):
    from mojo_align_env import getenv

    monkeypatch.setenv("MOJO_ALIGN_READ_BATCH", "16")
    monkeypatch.setenv("METHYLGRAPHER_MOJO_READ_BATCH", "99")
    assert getenv("READ_BATCH") == "16"


def test_getenv_falls_back_to_legacy(monkeypatch):
    from mojo_align_env import getenv

    monkeypatch.delenv("MOJO_ALIGN_READ_BATCH", raising=False)
    monkeypatch.setenv("METHYLGRAPHER_MOJO_READ_BATCH", "32")
    assert getenv("READ_BATCH") == "32"
