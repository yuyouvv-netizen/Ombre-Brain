"""Emotion coordinates describe experience and do not rank default breath."""
from datetime import datetime
from unittest.mock import MagicMock

import frontmatter
import pytest

import tools._runtime as rt
from tools.breath.surface import surface_default


class EchoDehydrator:
    async def dehydrate(self, content, meta=None):
        return content


class EmptyEmbedding:
    enabled = False


def install_runtime(bucket_mgr, decay_eng):
    rt.config = {"surfacing": {}}
    rt.bucket_mgr = bucket_mgr
    rt.decay_engine = decay_eng
    rt.dehydrator = EchoDehydrator()
    rt.embedding_engine = EmptyEmbedding()
    rt.logger = MagicMock()
    rt.fire_webhook = None
    rt.mark_op = None


def _pin_identical_timestamp(bucket_mgr, bucket_id: str, ts: str) -> None:
    fpath = bucket_mgr._find_bucket_file(bucket_id)
    post = frontmatter.load(fpath)
    post["created"] = ts
    post["last_active"] = ts
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(frontmatter.dumps(post))


@pytest.mark.asyncio
async def test_emotion_coordinates_do_not_break_surface_ties(
    bucket_mgr, decay_eng, monkeypatch,
):
    # Flatten score and time. Stable source order should remain; emotion must
    # not silently become a value judgement or ranking signal.
    monkeypatch.setattr(decay_eng, "calculate_score", lambda meta: 1.0)
    install_runtime(bucket_mgr, decay_eng)

    zero_id = await bucket_mgr.create(
        content="效价唤醒度都恰好为零的记忆", importance=5,
        valence=0.0, arousal=0.0,
    )
    low_id = await bucket_mgr.create(
        content="效价唤醒度真实偏低但不为零的记忆", importance=5,
        valence=0.4, arousal=0.25,  # av = 0.1
    )

    same_ts = datetime(2026, 1, 1).isoformat()
    _pin_identical_timestamp(bucket_mgr, zero_id, same_ts)
    _pin_identical_timestamp(bucket_mgr, low_id, same_ts)
    monkeypatch.setattr("tools.breath.surface.random.shuffle", lambda items: None)

    first = await surface_default(max_results=10, max_tokens=10000, tag_filter=[])
    first_order = first.index("效价唤醒度都恰好为零的记忆") < first.index(
        "效价唤醒度真实偏低但不为零的记忆"
    )

    await bucket_mgr.update(zero_id, valence=1.0, arousal=1.0)
    await bucket_mgr.update(low_id, valence=0.0, arousal=0.0)
    _pin_identical_timestamp(bucket_mgr, zero_id, same_ts)
    _pin_identical_timestamp(bucket_mgr, low_id, same_ts)
    second = await surface_default(max_results=10, max_tokens=10000, tag_filter=[])
    second_order = second.index("效价唤醒度都恰好为零的记忆") < second.index(
        "效价唤醒度真实偏低但不为零的记忆"
    )

    assert first_order == second_order
    assert "[权重:" not in first
    assert "[权重:" not in second
