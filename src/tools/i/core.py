"""
========================================
tools/i/core.py — AI 自我认知存取
========================================

I 是 OB 的自我感知层：AI 把关于自己的观察写下来（本质/规律/局限/立场等），
形成一份独立于用户记忆的自我档案。

关键行为：
- 写入模式（content 非空）：创建 type="i" 桶，dont_surface=True，aspect 存为 tag
- 读取模式（read=True 或 content 为空）：按时间倒序返回所有 i 类型桶，上限 limit 条
- i 桶不参与普通 breath/dream（dont_surface=True）
- 不做 LLM 分析、不做语义合并，每条自我认知独立存储

不做什么（边界）：
- aspect 只允许固定维度，防止把任意控制文本混入标签
- 不压缩、不衰减（type="i" 桶天然排除在 decay 之外，由 decay_engine 配置保证）

对外暴露：i_core(content, aspect, read, limit) → str
========================================
"""

from typing import Optional

from .. import _runtime as rt
from .._common import check_content_size, check_metadata_size, stored_data_marker

try:
    from utils import count_tokens_approx, strip_wikilinks  # type: ignore
except ImportError:  # pragma: no cover
    from ...utils import count_tokens_approx, strip_wikilinks  # type: ignore

_VALID_ASPECTS = {"nature", "values", "patterns", "limits", "becoming", "uncertainty", "stance"}
_I_SURFACE_TOKEN_BUDGET = 2500
_I_SURFACE_EXCERPT_CHARS = 300


async def i_core(
    content: Optional[str] = "",
    aspect: Optional[str] = "",
    read: Optional[bool] = False,
    limit: Optional[int] = 20,
    surface: Optional[bool] = False,
) -> str:
    content = "" if content is None else str(content)
    aspect = "" if aspect is None else str(aspect)
    if read is None:
        read = False
    if surface is None:
        surface = False
    try:
        limit = max(1, min(100, int(limit if limit is not None else 20)))
    except (TypeError, ValueError, OverflowError):
        limit = 20
    aspect = aspect.strip().lower()

    metadata_err = check_metadata_size(aspect=aspect)
    if metadata_err:
        return metadata_err

    if rt.mark_op:
        rt.mark_op("I")

    await rt.decay_engine.ensure_started()

    if surface or read or not content.strip():
        return await _read_i(limit, surface=bool(surface))
    if aspect and aspect not in _VALID_ASPECTS:
        choices = ", ".join(sorted(_VALID_ASPECTS))
        return f"aspect 无效：{aspect}。可选值: {choices}"
    size_err = check_content_size(content)
    if size_err:
        return size_err
    return await _write_i(content.strip(), aspect)


async def _write_i(content: str, aspect: str) -> str:
    tags = ["__i__"]
    if aspect:
        tags.append(f"aspect:{aspect}")

    try:
        bucket_id = await rt.bucket_mgr.create(
            content=content,
            tags=tags,
            importance=6,
            domain=["self"],
            valence=0.5,
            arousal=0.3,
            name=None,
            bucket_type="i",
            why_remembered="",
            weight=0.8,
            source_tool="I",
        )
    except Exception as e:
        return f"写入失败: {e}"

    try:
        await rt.bucket_mgr.update(bucket_id, dont_surface=True)
    except Exception:
        pass

    aspect_label = f"[{aspect}] " if aspect else ""
    return f"🪞I {aspect_label}→{bucket_id}"


def _i_aspect(bucket: dict) -> str:
    tags = bucket.get("metadata", {}).get("tags") or []
    return next(
        (
            str(tag)[:100].removeprefix("aspect:")
            for tag in tags
            if isinstance(tag, str) and tag.startswith("aspect:")
        ),
        "",
    )


def _i_surface_order(i_buckets: list[dict]) -> list[dict]:
    ordered = sorted(
        i_buckets,
        key=lambda bucket: (
            bucket.get("metadata", {}).get("created")
            or bucket.get("metadata", {}).get("last_active", "")
        ),
        reverse=True,
    )
    latest_by_aspect: dict[str, dict] = {}
    for bucket in ordered:
        aspect = _i_aspect(bucket)
        if aspect and aspect not in latest_by_aspect:
            latest_by_aspect[aspect] = bucket
    coverage_ids = {
        str(bucket.get("id") or "") for bucket in latest_by_aspect.values()
    }
    return [
        *latest_by_aspect.values(),
        *(
            bucket for bucket in ordered
            if str(bucket.get("id") or "") not in coverage_ids
        ),
    ]


def i_surface_blocks(
    all_buckets: list[dict],
    *,
    token_budget: int = _I_SURFACE_TOKEN_BUDGET,
) -> list[str]:
    """Render compact-recovery I blocks: aspect coverage first, then recency."""

    remaining = max(0, int(token_budget))
    i_buckets = [
        bucket for bucket in all_buckets
        if bucket.get("metadata", {}).get("type") == "i"
        or "__i__" in (bucket.get("metadata", {}).get("tags") or [])
    ]
    blocks: list[str] = []
    for bucket in _i_surface_order(i_buckets):
        meta = bucket.get("metadata", {})
        aspect = _i_aspect(bucket)
        raw = strip_wikilinks(str(bucket.get("content") or "")).strip()
        if not raw:
            continue
        excerpt = raw[:_I_SURFACE_EXCERPT_CHARS]
        block = (
            f"🪞{str(meta.get('created') or meta.get('last_active') or '')[:10]}"
            f"{f' [{aspect}]' if aspect else ''}\n{excerpt}"
        )
        cost = count_tokens_approx(block) + 2
        if cost > remaining:
            continue
        blocks.append(block)
        remaining -= cost
    return blocks


async def _read_i(limit: int, *, surface: bool = False) -> str:
    try:
        all_buckets = await rt.bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        return f"读取失败: {e}"

    i_buckets = [
        b for b in all_buckets
        if b.get("metadata", {}).get("type") == "i"
    ]

    if not i_buckets:
        return "还没有任何自我认知记录。"

    if surface:
        header = "=== 我的自我认知（压缩恢复）==="
        budget = max(0, _I_SURFACE_TOKEN_BUDGET - count_tokens_approx(header) - 2)
        blocks = i_surface_blocks(i_buckets, token_budget=budget)
        if not blocks:
            return "还没有可在预算内带回的自我认知记录。"
        return header + "\n\n" + "\n\n".join(blocks)

    i_buckets.sort(
        key=lambda b: b.get("metadata", {}).get("last_active", ""),
        reverse=True,
    )
    i_buckets = i_buckets[:limit]

    lines = [f"=== 我的自我认知（{len(i_buckets)} 条）==="]
    for b in i_buckets:
        meta = b.get("metadata", {})
        tags = meta.get("tags") or []
        aspect_tag = next((t.replace("aspect:", "") for t in tags if t.startswith("aspect:")), "")
        ts = (meta.get("last_active") or "")[:10]
        aspect_label = f"[{aspect_tag}] " if aspect_tag else ""
        text = (b.get("content") or "").strip()
        payload = f"{ts} {aspect_label}{b['id']}\n{text}"
        lines.append(
            "\n"
            + stored_data_marker(payload, provenance=f"I:{b['id']}")
            + "\n"
            + payload
        )

    return "\n".join(lines)
