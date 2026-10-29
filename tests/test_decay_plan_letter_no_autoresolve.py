"""decay_engine 不得碰 plan/letter 回归测试。

找茬会话发现的 bug：run_decay_cycle() 的跳过名单只写了
("permanent", "feel", "i")，漏了 plan/letter。calculate_score() 确实对
plan/letter 恒定返回高于阈值的分数（不会被归档），但同一循环里跑在打分
之前的「自动结案」分支（重要度≤4 + 超过30天未激活 → resolved=True）不看
type，会直接把地位是"生命周期只能由 status 字段驱动"的 plan、和"承诺永久
原样保留"的 letter 一起标成 resolved——绕开了它们各自的真实生命周期规则。

修复：跳过名单加上 "plan"、"letter"，跟 tools/_common.py 的
cascade_plan_resolved_to_buckets() 用的排除逻辑保持一致。
"""
import frontmatter as fm
import pytest


def _backdate(bucket_mgr, bucket_id: str, days_ago: int) -> None:
    """把桶的 created/last_active 都改成 N 天前，制造「超期未激活」的条件。"""
    from datetime import datetime, timedelta

    fpath = bucket_mgr._find_bucket_file(bucket_id)
    post = fm.load(fpath)
    old_ts = (datetime.now() - timedelta(days=days_ago)).isoformat()
    post["created"] = old_ts
    post["last_active"] = old_ts
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(fm.dumps(post))


@pytest.mark.asyncio
async def test_decay_cycle_never_auto_resolves_plan(bucket_mgr, decay_eng):
    bid = await bucket_mgr.create(
        content="记得帮她把这件事办完", bucket_type="plan", importance=3,
    )
    _backdate(bucket_mgr, bid, days_ago=60)

    await decay_eng.run_decay_cycle()

    plan = await bucket_mgr.get(bid)
    assert plan["metadata"].get("resolved") is not True, (
        "plan 的生命周期只能由 status 字段驱动，decay 的自动结案不该碰它"
    )
    assert plan["metadata"]["type"] == "plan"


@pytest.mark.asyncio
async def test_decay_cycle_never_auto_resolves_letter(bucket_mgr, decay_eng):
    bid = await bucket_mgr.create(
        content="给未来自己的一封信", bucket_type="letter", importance=3,
    )
    _backdate(bucket_mgr, bid, days_ago=60)

    await decay_eng.run_decay_cycle()

    letter = await bucket_mgr.get(bid)
    assert letter["metadata"].get("resolved") is not True, (
        "letter 承诺永久原样保留，decay 的自动结案不该碰它"
    )
    assert letter["metadata"]["type"] == "letter"


@pytest.mark.asyncio
async def test_decay_cycle_preserves_ordinary_dynamic_bucket(bucket_mgr, decay_eng):
    """普通桶也必须保持原文、状态和位置，不能因为时间自动结案。"""
    bid = await bucket_mgr.create(content="一件早就不重要的小事", importance=3)
    _backdate(bucket_mgr, bid, days_ago=60)

    stats = await decay_eng.run_decay_cycle()

    ordinary = await bucket_mgr.get(bid)
    assert ordinary["metadata"].get("resolved") is not True
    assert ordinary["metadata"]["type"] == "dynamic"
    assert ordinary["content"] == "一件早就不重要的小事"
    assert stats["auto_resolved"] == 0
    assert stats["archived"] == 0


@pytest.mark.asyncio
async def test_old_memory_keeps_bytes_and_can_surface_with_result_limit(bucket_mgr, decay_eng, monkeypatch):
    from pathlib import Path
    from unittest.mock import MagicMock
    import tools._runtime as rt
    from tools.breath.surface import surface_default

    old = await bucket_mgr.create(content="放了十年的重要原文", importance=7)
    _backdate(bucket_mgr, old, days_ago=3650)
    path = Path(bucket_mgr._find_bucket_file(old))
    before = path.read_bytes()
    newer = await bucket_mgr.create(content="今天的一条普通记录", importance=3)
    archived = await bucket_mgr.create(content="手动归档原文", importance=5)
    await bucket_mgr.archive(archived)
    # 即使保留旧部署里的极端参数，也不能自动改变状态。
    decay_eng.decay_lambda = 99
    decay_eng.threshold = 9999
    for _ in range(2):
        stats = await decay_eng.run_decay_cycle()
        assert stats["archived"] == stats["auto_resolved"] == 0
    assert path.read_bytes() == before
    assert (await bucket_mgr.get(archived))["metadata"]["type"] == "archived"
    assert decay_eng.calculate_score((await bucket_mgr.get(old))["metadata"]) == 7.0
    assert decay_eng.calculate_score((await bucket_mgr.get(newer))["metadata"]) == 3.0

    for key, value in {
        "config": {"surfacing": {}}, "bucket_mgr": bucket_mgr, "decay_engine": decay_eng,
        "logger": MagicMock(), "mark_op": None, "fire_webhook": None,
    }.items():
        monkeypatch.setattr(rt, key, value)
    monkeypatch.setattr("tools.breath.surface.random.random", lambda: 1.0)
    result = await surface_default(max_results=1, max_tokens=10000, tag_filter=[])
    assert "放了十年的重要原文" in result
    assert "今天的一条普通记录" not in result
    assert "手动归档原文" not in result
