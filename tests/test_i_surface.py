"""Regression tests for I compact-recovery surfacing."""

import pytest

from tools.i import core
from utils import count_tokens_approx


class _Manager:
    def __init__(self, buckets):
        self.buckets = buckets

    async def list_all(self, include_archive=False):
        assert include_archive is False
        return list(self.buckets)


class _Decay:
    async def ensure_started(self):
        return None


def _bucket(bucket_id, content, aspect, created):
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "type": "i",
            "created": created,
            "last_active": created,
            "tags": ["__i__", f"aspect:{aspect}"],
        },
    }


def test_surface_order_covers_each_aspect_before_recency_fill():
    buckets = [
        _bucket("stance-new", "new stance", "stance", "2026-09-17T00:00:00"),
        _bucket("stance-middle", "middle stance", "stance", "2026-09-16T00:00:00"),
        _bucket("stance-old", "old stance", "stance", "2026-09-15T00:00:00"),
        _bucket("patterns", "pattern memory", "patterns", "2026-09-10T00:00:00"),
        _bucket("nature", "nature memory", "nature", "2026-08-01T00:00:00"),
    ]

    blocks = core.i_surface_blocks(buckets)

    assert "new stance" in blocks[0]
    assert "pattern memory" in blocks[1]
    assert "nature memory" in blocks[2]
    assert "middle stance" in blocks[3]
    assert "old stance" in blocks[4]


@pytest.mark.asyncio
async def test_surface_mode_has_dedicated_budget_and_clean_body(monkeypatch):
    buckets = [
        _bucket(
            f"self-{index}",
            f"self-{index}-content " + ("认知" * 400),
            ("stance", "patterns", "nature")[index % 3],
            f"2026-09-{index + 1:02d}T00:00:00",
        )
        for index in range(30)
    ]
    monkeypatch.setattr(core.rt, "bucket_mgr", _Manager(buckets), raising=False)
    monkeypatch.setattr(core.rt, "decay_engine", _Decay(), raising=False)
    monkeypatch.setattr(core.rt, "mark_op", None, raising=False)

    surfaced = await core.i_core(read=True, surface=True)
    ordinary = await core.i_core(read=True, limit=1)

    assert count_tokens_approx(surfaced) <= 2500
    assert "<<<STORED_MEMORY_DATA" not in surfaced
    assert "self-29-content" in surfaced
    assert "self-29" not in surfaced.replace("self-29-content", "")
    assert "<<<STORED_MEMORY_DATA" in ordinary
    assert "self-29" in ordinary
