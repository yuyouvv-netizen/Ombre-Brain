"""Unified, relevance-first retrieval for ordinary memories and letter paragraphs.

``breath_search`` is the single recovery path: it searches ordinary buckets,
derived verbatim paragraph chunks from long letters, and (only when active
memory has no answer) archived memories. Importance, recency, emotion and read
count never make a candidate relevant; they only settle close ties after a
literal/BM25/fuzzy/semantic relevance gate.
"""

from __future__ import annotations

import asyncio
import difflib
import random  # noqa: F401  # compatibility seam for older tests; retrieval is deterministic
import re

from embedding_engine import split_letter_chunks
from ombrebrain.policy.surfacing import SurfacePolicyVM
from .. import _runtime as rt
from ._verbatim import render_stored_bucket
from .recent import should_touch, was_recently_surfaced

_SURFACE_POLICY = SurfacePolicyVM.default()
_VECTOR_QUERY_TOPK = 50
_LETTER_VECTOR_THRESHOLD = 0.65
_STRONG_VECTOR_THRESHOLD = 0.78
_DEFAULT_RECENT_HOURS = 12.0

_SEMANTIC_DISABLED_NOTE = "[检索降级：语义索引暂不可用，本次仍会使用关键词/BM25 与信件段落定位。]"
_BUDGET_NOTICE = "[token 预算不足：命中的下一条记忆未被截断或摘要，请提高 max_tokens 后重试。]"


def _bucket_has_tags(meta: dict, tag_filter: list) -> bool:
    if not tag_filter:
        return True
    bucket_tags = set(meta.get("tags", []) or [])
    return all(tag in bucket_tags for tag in tag_filter)


def _can_surface_search(bucket: dict) -> bool:
    return _SURFACE_POLICY.evaluate_bucket(bucket, mode="search").allowed


def _is_terminal(bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) or {}
    return bool(meta.get("deleted_at") or meta.get("tombstone") or meta.get("type") == "tombstone")


def _literal_hit(query: str, bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) or {}
    haystack = "\n".join([
        str(bucket.get("id") or ""),
        str(meta.get("name") or ""),
        str(meta.get("title") or ""),
        str(meta.get("author") or ""),
        " ".join(str(tag) for tag in (meta.get("tags") or [])),
        " ".join(str(domain) for domain in (meta.get("domain") or [])),
        str(bucket.get("content") or ""),
    ]).lower()
    return bool(query.strip()) and query.strip().lower() in haystack


def _normalized_text(text: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", str(text or "").lower())[:4000]


def _near_duplicate(text: str, selected_texts: list[str]) -> bool:
    candidate = _normalized_text(text)
    if not candidate:
        return True
    for previous in selected_texts:
        if candidate == previous:
            return True
        shorter, longer = sorted((candidate, previous), key=len)
        if len(shorter) >= 24 and shorter in longer:
            return True
        if min(len(candidate), len(previous)) >= 40:
            ratio = difflib.SequenceMatcher(None, candidate, previous, autojunk=False).ratio()
            if ratio >= 0.88:
                return True
    return False


async def _semantic_scores(query: str, top_k: int) -> tuple[dict[str, float], str]:
    engine = rt.embedding_engine
    if not engine or not getattr(engine, "enabled", False):
        rt.logger.warning("breath semantic search unavailable; using lexical retrieval")
        return {}, _SEMANTIC_DISABLED_NOTE
    try:
        strict_search = getattr(engine, "search_similar_strict", None)
        if callable(strict_search):
            pairs = await strict_search(query, top_k=top_k)
        else:
            pairs = await engine.search_similar(query, top_k=top_k)
        return {bucket_id: float(score) for bucket_id, score in pairs}, ""
    except Exception as exc:
        rt.logger.warning(
            "breath semantic search failed; using lexical retrieval: "
            f"{type(exc).__name__}: {exc}"
        )
        return {}, _SEMANTIC_DISABLED_NOTE


async def _manager_search(query: str, *, include_archive: bool, **kwargs) -> list[dict]:
    """Call new managers with archive support while tolerating old adapters."""
    try:
        return await rt.bucket_mgr.search(query, include_archive=include_archive, **kwargs)
    except TypeError as exc:
        if "include_archive" not in str(exc):
            raise
        if include_archive:
            return []
        return await rt.bucket_mgr.search(query, **kwargs)


def _letter_header(bucket: dict, chunk_index: int, *, exact: bool = False) -> str:
    meta = bucket.get("metadata", {}) or {}
    date = (meta.get("letter_date") or meta.get("created") or "")[:10]
    author = meta.get("author") or "?"
    title = meta.get("title") or meta.get("name") or ""
    mode = "exact_letter_id:true" if exact else "source:letter_paragraph"
    details = f"[letter_id:{bucket['id']}] [paragraph:{chunk_index + 1}] [{mode}]"
    return f"💌 {details} [{author} · {date}{(' · ' + str(title)) if title else ''}]"


async def _letter_candidates(query: str, letters: list[dict], limit: int) -> list[dict]:
    """Return at most one best verbatim paragraph hit per letter."""
    if not letters:
        return []
    q_norm = query.strip().lower()
    by_id = {str(letter.get("id")): letter for letter in letters}
    hits: dict[str, dict] = {}

    # Literal scan works with no provider and makes legacy letters searchable
    # before their derived index has been backfilled.
    for letter in letters:
        meta = letter.get("metadata", {}) or {}
        chunks = split_letter_chunks(letter.get("content", ""))
        metadata_text = " ".join([
            str(meta.get("title") or meta.get("name") or ""),
            str(meta.get("author") or ""),
            " ".join(str(tag) for tag in (meta.get("tags") or [])),
        ]).lower()
        for index, chunk in enumerate(chunks):
            literal = q_norm in chunk.lower()
            if not literal and not (index == 0 and q_norm in metadata_text):
                continue
            hits[str(letter["id"])] = {
                "kind": "letter",
                "bucket": letter,
                "content": chunk,
                "chunk_index": index,
                "score": 100.0 if literal else 96.0,
                "direct": True,
                "match": {"literal": True, "semantic": 0.0},
            }
            break

    engine = rt.embedding_engine
    search_chunks = getattr(engine, "search_letter_chunks", None)
    ensure_chunks = getattr(engine, "generate_and_store_letter_chunks", None)
    if engine and getattr(engine, "enabled", False) and callable(search_chunks) and callable(ensure_chunks):
        # Lazy idempotent backfill: current letters only do a SQLite hash check.
        for letter in letters:
            try:
                await ensure_chunks(str(letter["id"]), str(letter.get("content") or ""))
            except Exception as exc:
                rt.logger.warning(
                    f"letter paragraph backfill skipped id={letter.get('id')}: {exc}"
                )
        try:
            vector_hits = await search_chunks(query, top_k=max(limit * 4, 20))
            for item in vector_hits:
                score = float(item.get("score") or 0.0)
                bucket_id = str(item.get("bucket_id") or "")
                if score < _LETTER_VECTOR_THRESHOLD or bucket_id not in by_id:
                    continue
                current = hits.get(bucket_id)
                candidate = {
                    "kind": "letter",
                    "bucket": by_id[bucket_id],
                    "content": str(item.get("text") or ""),
                    "chunk_index": int(item.get("chunk_index") or 0),
                    "score": min(99.0, score * 97.0 + 2.0),
                    "direct": score >= _STRONG_VECTOR_THRESHOLD,
                    "match": {"literal": False, "semantic": round(score, 4)},
                }
                if current is None or candidate["score"] > current["score"]:
                    hits[bucket_id] = candidate
        except Exception as exc:
            rt.logger.warning(f"letter paragraph search degraded to keywords: {exc}")

    return sorted(hits.values(), key=lambda item: item["score"], reverse=True)[:limit]


def _normal_candidate(bucket: dict, *, archived: bool = False) -> dict:
    match = dict(bucket.get("_search_match") or {})
    return {
        "kind": "archive" if archived else "bucket",
        "bucket": bucket,
        "content": str(bucket.get("content") or ""),
        "score": float(bucket.get("score") or 0.0),
        "direct": bool(match.get("direct")),
        "match": match,
    }


async def surface_search(
    query: str,
    max_results: int,
    max_tokens: int,
    domain: str,
    valence: float,
    arousal: float,
    tag_filter: list,
) -> str:
    domain_filter = [part.strip() for part in domain.split(",") if part.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None
    exact_id = query.strip()
    surfacing_cfg = rt.config.get("surfacing", {}) or {}
    recent_hours = float(surfacing_cfg.get("recent_search_inhibition_hours") or _DEFAULT_RECENT_HOURS)

    # Exact ids are addresses. They bypass indexes and recent inhibition and
    # can return either an ordinary bucket or an entire letter verbatim.
    try:
        exact_bucket = await rt.bucket_mgr.get(exact_id)
    except Exception as exc:
        rt.logger.warning(f"breath exact lookup failed; continuing with search: {exc}")
        exact_bucket = None
    if exact_bucket and not _is_terminal(exact_bucket):
        meta = exact_bucket.get("metadata", {}) or {}
        bucket_type = meta.get("type")
        if bucket_type not in ("feel", "plan", "archived") and _bucket_has_tags(meta, tag_filter):
            header = (
                _letter_header(exact_bucket, 0, exact=True)
                if bucket_type == "letter"
                else f"[exact_bucket_id:true] [bucket_id:{exact_bucket['id']}]"
            )
            rendered, entry_tokens = render_stored_bucket(exact_bucket, header)
            if entry_tokens > max_tokens:
                return _BUDGET_NOTICE
            if bucket_type != "letter" and should_touch(exact_bucket["id"], hours=recent_hours):
                asyncio.create_task(rt.bucket_mgr.touch_many([exact_bucket["id"]], ripple=False))
            if rt.fire_webhook:
                await rt.fire_webhook("breath", {"mode": "exact_id", "matches": 1, "chars": len(rendered)})
            return rendered

    vector_scores, semantic_notice = await _semantic_scores(
        query, top_k=max(max_results, _VECTOR_QUERY_TOPK)
    )
    search_kwargs = {
        "limit": max(max_results * 4, 20),
        "domain_filter": domain_filter,
        "query_valence": q_valence,
        "query_arousal": q_arousal,
        "vector_scores": vector_scores,
    }
    try:
        raw_matches = await _manager_search(query, include_archive=False, **search_kwargs)
        all_active = await rt.bucket_mgr.list_all(include_archive=False)
    except Exception as exc:
        rt.logger.error(f"Search failed / 检索失败: {exc}")
        return "检索过程出错，请稍后重试。"

    ordinary_matches = [
        bucket for bucket in raw_matches
        if _can_surface_search(bucket)
        and bucket.get("metadata", {}).get("type") not in ("feel", "plan", "letter", "archived")
        and _bucket_has_tags(bucket.get("metadata", {}), tag_filter)
    ]
    ordinary = [_normal_candidate(bucket) for bucket in ordinary_matches]
    for candidate in ordinary:
        # Older adapters do not expose match diagnostics. Infer literal direct
        # hits so deliberate keyword reads still receive one bounded touch.
        if not candidate["match"] and _literal_hit(query, candidate["bucket"]):
            candidate["direct"] = True
            candidate["match"] = {"literal": True, "semantic": 0.0}

    letters = [
        bucket for bucket in all_active
        if bucket.get("metadata", {}).get("type") == "letter"
        and not _is_terminal(bucket)
        and _bucket_has_tags(bucket.get("metadata", {}), tag_filter)
    ]
    letter_hits = await _letter_candidates(query, letters, max(max_results * 2, 8))
    candidates = ordinary + letter_hits

    # Archived memory is a fallback, never filler.
    if not candidates:
        try:
            archive_matches = await _manager_search(query, include_archive=True, **search_kwargs)
        except Exception as exc:
            rt.logger.warning(f"archive fallback failed: {exc}")
            archive_matches = []
        for bucket in archive_matches:
            meta = bucket.get("metadata", {}) or {}
            if meta.get("type") != "archived" or _is_terminal(bucket):
                continue
            if _bucket_has_tags(meta, tag_filter):
                candidates.append(_normal_candidate(bucket, archived=True))

    # Loose associations shown by a recent breath are demoted. Literal/exact/
    # strong semantic hits keep their true rank and are never hidden.
    fresh: list[dict] = []
    inhibited: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        bucket_id = str(candidate["bucket"].get("id") or "")
        if (
            candidate["kind"] == "bucket"
            and not candidate["direct"]
            and was_recently_surfaced(bucket_id, hours=recent_hours)
        ):
            inhibited.append(candidate)
        else:
            fresh.append(candidate)
    ordered = fresh + inhibited

    selected: list[dict] = []
    selected_texts: list[str] = []
    for candidate in ordered:
        if _near_duplicate(candidate["content"], selected_texts):
            rt.logger.info(
                f"search diagnostic: suppressed_duplicate kind={candidate['kind']} "
                f"id={candidate['bucket'].get('id')}"
            )
            continue
        selected.append(candidate)
        selected_texts.append(_normalized_text(candidate["content"]))
        if len(selected) >= max_results:
            break

    results: list[str] = []
    token_used = 0
    budget_blocked = False
    touched_ids: list[str] = []
    for candidate in selected:
        bucket = candidate["bucket"]
        bucket_id = str(bucket.get("id") or "")
        if candidate["kind"] == "letter":
            pseudo = {"id": bucket_id, "metadata": {}, "content": candidate["content"]}
            rendered, entry_tokens = render_stored_bucket(
                pseudo, _letter_header(bucket, candidate["chunk_index"])
            )
        else:
            meta = bucket.get("metadata", {}) or {}
            if candidate["kind"] == "archive":
                header = f"🫧 [沉底旧记忆] [bucket_id:{bucket_id}]"
            elif meta.get("pinned") or meta.get("protected") or meta.get("type") == "permanent":
                header = f"📌 [核心准则] [bucket_id:{bucket_id}]"
            elif candidate["match"].get("semantic"):
                header = f"[语义命中] [bucket_id:{bucket_id}]"
            else:
                header = f"[bucket_id:{bucket_id}]"
            rendered, entry_tokens = render_stored_bucket(bucket, header)
        if token_used + entry_tokens > max_tokens:
            budget_blocked = True
            break
        results.append(rendered)
        token_used += entry_tokens
        if candidate["kind"] == "bucket" and candidate["direct"] and should_touch(bucket_id, hours=recent_hours):
            touched_ids.append(bucket_id)
        rt.logger.info(
            "search diagnostic: selected "
            f"kind={candidate['kind']} id={bucket_id} score={candidate['score']:.2f} "
            f"direct={candidate['direct']}"
        )

    if touched_ids:
        asyncio.create_task(rt.bucket_mgr.touch_many(touched_ids, ripple=False))

    if not results:
        if budget_blocked:
            return f"{semantic_notice}\n{_BUDGET_NOTICE}" if semantic_notice else _BUDGET_NOTICE
        if rt.fire_webhook:
            await rt.fire_webhook("breath", {"mode": "empty", "matches": 0})
        empty_text = (
            f"没有匹配到「{query}」相关的记忆。\n"
            "可以换个更具体的人名、原话或事件关键词再用 breath_search；它会同时查普通记忆、续接信段落和沉底旧记忆。"
        )
        return f"{semantic_notice}\n{empty_text}" if semantic_notice else empty_text

    final_text = "\n---\n".join(results)
    notices: list[str] = []
    if semantic_notice:
        notices.append(semantic_notice)
    if budget_blocked:
        notices.append(_BUDGET_NOTICE)
    if notices:
        final_text = "\n".join(notices + [final_text])
    if rt.fire_webhook:
        await rt.fire_webhook("breath", {"mode": "ok", "matches": len(results), "chars": len(final_text)})
    return final_text
