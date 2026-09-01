"""Unified, relevance-first retrieval for memories and letter paragraphs.

``breath_search`` is the single recovery path. It searches active ordinary
memories and derived verbatim letter paragraphs first, then consults archived
memory only when the active set has no answer. Relevance admits candidates;
stable importance and recency may only settle close ties afterward.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import random  # noqa: F401  # compatibility seam; search itself stays deterministic
import re
from datetime import datetime, time

from embedding_engine import split_letter_chunks
from ombrebrain.policy.surfacing import SurfacePolicyVM
from utils import parse_iso_datetime

from .. import _runtime as rt
from ._verbatim import render_stored_bucket
from .recent import should_touch, was_recently_surfaced

_SURFACE_POLICY = SurfacePolicyVM.default()
_VECTOR_QUERY_TOPK = 50
_LETTER_VECTOR_THRESHOLD = 0.65
_STRONG_VECTOR_THRESHOLD = 0.78
_DEFAULT_RECENT_HOURS = 12.0

_SEMANTIC_DISABLED_NOTE = (
    "[检索降级：语义索引暂不可用，本次仍使用关键词/BM25 与信件段落定位。]"
)
_BUDGET_NOTICE = (
    "[token 预算不足：命中的下一条记忆未被截断或摘要，请提高 max_tokens 后重试。]"
)


def _bucket_has_tags(meta: dict, tag_filter: list) -> bool:
    if not tag_filter:
        return True
    bucket_tags = set(meta.get("tags", []) or [])
    return all(tag in bucket_tags for tag in tag_filter)


def _bucket_domains(meta: dict) -> set[str]:
    raw = meta.get("domain") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(item).strip().lower() for item in raw if str(item).strip()}


def _domain_matches(meta: dict, domain_filter: list[str] | None) -> bool:
    if not domain_filter:
        return True
    wanted = {str(item).strip().lower() for item in domain_filter}
    return bool(_bucket_domains(meta).intersection(wanted))


def _can_surface_search(bucket: dict) -> bool:
    return _SURFACE_POLICY.evaluate_bucket(bucket, mode="search").allowed


def _is_archived(bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) or {}
    return (
        str(meta.get("type") or "").strip().lower() == "archived"
        or bool(meta.get("deleted_at"))
        or bool(meta.get("tombstone"))
    )


def _literal_hit(query: str, bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) or {}
    haystack = "\n".join([
        str(bucket.get("id") or ""),
        str(meta.get("name") or ""),
        str(meta.get("title") or ""),
        str(meta.get("author") or ""),
        " ".join(str(tag) for tag in (meta.get("tags") or [])),
        " ".join(sorted(_bucket_domains(meta))),
        str(bucket.get("content") or ""),
    ]).lower()
    needle = query.strip().lower()
    if not needle:
        return False
    return needle in haystack or _normalized_text(needle) in _normalized_text(haystack)


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
            ratio = difflib.SequenceMatcher(
                None, candidate, previous, autojunk=False
            ).ratio()
            if ratio >= 0.88:
                return True
    return False


def _parse_date_bound(value: str, *, upper: bool) -> datetime | None:
    """Parse a created-time boundary; a date-only upper bound includes its day."""
    raw = value.strip()
    if not raw:
        return None
    parsed = parse_iso_datetime(raw)
    if len(raw) == 10:
        day = parsed.date()
        return datetime.combine(day, time.max if upper else time.min)
    return parsed


def _bucket_in_created_range(
    bucket: dict,
    created_from: datetime | None,
    created_to: datetime | None,
) -> bool:
    if created_from is None and created_to is None:
        return True
    raw_created = str((bucket.get("metadata", {}) or {}).get("created") or "").strip()
    if not raw_created:
        return False
    try:
        created = parse_iso_datetime(raw_created)
    except (TypeError, ValueError):
        return False
    if created_from is not None and created < created_from:
        return False
    if created_to is not None and created > created_to:
        return False
    return True


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


def _semantic_diagnostics(
    query: str,
    vector_scores: dict[str, float],
    semantic_notice: str,
) -> dict:
    """Collect reconstructable index state without logging the query text."""
    engine_status: dict = {}
    status_reader = getattr(rt.embedding_engine, "status", None)
    if callable(status_reader):
        try:
            raw_status = status_reader()
            if isinstance(raw_status, dict):
                engine_status = dict(raw_status)
        except Exception as exc:
            engine_status = {"status_error": f"{type(exc).__name__}: {exc}"}

    outbox_status: dict = {}
    status_reader = getattr(getattr(rt, "embedding_outbox", None), "status", None)
    if callable(status_reader):
        try:
            raw_status = status_reader()
            if isinstance(raw_status, dict):
                outbox_status = dict(raw_status)
        except Exception as exc:
            outbox_status = {"status_error": f"{type(exc).__name__}: {exc}"}

    ranked = sorted(vector_scores.items(), key=lambda item: item[1], reverse=True)
    return {
        "query_hash": hashlib.sha256(
            query.encode("utf-8", errors="replace")
        ).hexdigest()[:12],
        "semantic_available": not bool(semantic_notice),
        "vector_candidates": len(vector_scores),
        "vector_top": [
            {"bucket_id": bucket_id, "score": round(score, 6)}
            for bucket_id, score in ranked[:5]
        ],
        "engine": {
            key: engine_status.get(key)
            for key in (
                "enabled", "backend", "model", "vector_dim",
                "embedding_count", "status_error",
            )
            if key in engine_status
        },
        "outbox": {
            key: outbox_status.get(key)
            for key in (
                "running", "provider_ready", "pending", "retrying",
                "last_success", "last_error", "status_error",
            )
            if key in outbox_status
        },
    }


async def _manager_search(query: str, *, include_archive: bool, **kwargs) -> list[dict]:
    """Call current managers with archive support while tolerating old adapters."""
    try:
        return await rt.bucket_mgr.search(
            query, include_archive=include_archive, **kwargs
        )
    except TypeError as exc:
        if "include_archive" not in str(exc):
            raise
        if include_archive:
            return []
        return await rt.bucket_mgr.search(query, **kwargs)


def _letter_footer(bucket: dict, chunk_index: int, *, exact: bool = False) -> str:
    meta = bucket.get("metadata", {}) or {}
    date = str(meta.get("letter_date") or meta.get("created") or "")[:10]
    author = meta.get("author") or "?"
    title = meta.get("title") or meta.get("name") or ""
    mode = "exact_letter_id:true" if exact else "source:letter_paragraph"
    source = f"{author} · {date}{(' · ' + str(title)) if title else ''}"
    return (
        f"💌 [{mode}] [paragraph:{chunk_index + 1}] [{source}] "
        f"[letter_id:{bucket['id']}]"
    )


def _archive_footer(bucket_id: str, *, exact: bool = False) -> str:
    mode = "exact_archive_id:true" if exact else "沉底旧记忆"
    return (
        f"🫧 [{mode}] [状态:已退出日常记忆，原文仍保留] "
        "[若决定恢复:使用该 bucket_id 调用 trace(restore=True)] "
        f"[bucket_id:{bucket_id}]"
    )


async def _letter_candidates(
    query: str,
    letters: list[dict],
    limit: int,
) -> list[dict]:
    """Return at most one best verbatim paragraph hit per letter."""
    if not letters:
        return []
    q_norm = query.strip().lower()
    by_id = {str(letter.get("id")): letter for letter in letters}
    hits: dict[str, dict] = {}

    # Literal scan works with no provider and covers letters not yet backfilled.
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
    if (
        engine
        and getattr(engine, "enabled", False)
        and callable(search_chunks)
        and callable(ensure_chunks)
    ):
        for letter in letters:
            try:
                await ensure_chunks(
                    str(letter["id"]), str(letter.get("content") or "")
                )
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
                candidate = {
                    "kind": "letter",
                    "bucket": by_id[bucket_id],
                    "content": str(item.get("text") or ""),
                    "chunk_index": int(item.get("chunk_index") or 0),
                    "score": min(99.0, score * 97.0 + 2.0),
                    "direct": score >= _STRONG_VECTOR_THRESHOLD,
                    "match": {"literal": False, "semantic": round(score, 4)},
                }
                current = hits.get(bucket_id)
                if current is None or candidate["score"] > current["score"]:
                    hits[bucket_id] = candidate
        except Exception as exc:
            rt.logger.warning(f"letter paragraph search degraded to keywords: {exc}")

    return sorted(hits.values(), key=_candidate_rank, reverse=True)[:limit]


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


def _candidate_rank(candidate: dict) -> tuple[int, float]:
    """Literal evidence always outranks associative retriever scores."""
    match = candidate.get("match") or {}
    return (
        1 if match.get("literal") else 0,
        float(candidate.get("score") or 0.0),
    )


async def surface_search(
    query: str,
    max_results: int,
    max_tokens: int,
    domain: str,
    valence: float,
    arousal: float,
    tag_filter: list,
    date_from: str = "",
    date_to: str = "",
) -> str:
    domain_filter = [part.strip() for part in domain.split(",") if part.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None
    try:
        created_from = _parse_date_bound(date_from, upper=False)
        created_to = _parse_date_bound(date_to, upper=True)
    except (TypeError, ValueError):
        return "日期格式无效，请使用 YYYY-MM-DD 或 ISO 8601 时间。"
    if created_from and created_to and created_from > created_to:
        return "date_from 不能晚于 date_to。"

    surfacing_cfg = rt.config.get("surfacing", {}) or {}
    recent_hours = float(
        surfacing_cfg.get("recent_search_inhibition_hours")
        or _DEFAULT_RECENT_HOURS
    )

    # Footprint is never loaded by normal breath or active search. It is read
    # lazily only when an archive hit must be classified by its original kind.
    footprint_snapshot = None
    footprint_loaded = False

    def _archived_original_kind(bucket: dict) -> str:
        nonlocal footprint_snapshot, footprint_loaded
        meta = bucket.get("metadata", {}) or {}
        explicit = str(meta.get("original_type") or "").strip().lower()
        if explicit:
            return explicit
        if not footprint_loaded:
            footprint_loaded = True
            try:
                footprint_snapshot = rt.bucket_mgr.footprint_snapshot()
            except Exception as exc:
                rt.logger.warning(f"Footprint classification unavailable: {exc}")
        if footprint_snapshot is None:
            return "dynamic"
        return footprint_snapshot.original_kind(
            str(bucket.get("id") or ""), meta
        )

    # Full ids are addresses and bypass indexes and short-term inhibition.
    exact_id = query.strip()
    try:
        exact_reader = getattr(rt.bucket_mgr, "get_including_archive", None)
        exact_bucket = (
            await exact_reader(exact_id)
            if callable(exact_reader)
            else await rt.bucket_mgr.get(exact_id)
        )
    except Exception as exc:
        rt.logger.warning(f"breath exact lookup failed; continuing with search: {exc}")
        exact_bucket = None
    if exact_bucket:
        meta = exact_bucket.get("metadata", {}) or {}
        bucket_type = str(meta.get("type") or "dynamic").strip().lower()
        in_scope = (
            _bucket_has_tags(meta, tag_filter)
            and _domain_matches(meta, domain_filter)
            and _bucket_in_created_range(exact_bucket, created_from, created_to)
        )
        if _is_archived(exact_bucket):
            original_kind = _archived_original_kind(exact_bucket)
            if original_kind not in ("feel", "plan", "letter") and in_scope:
                rendered, entry_tokens = render_stored_bucket(
                    exact_bucket,
                    _archive_footer(exact_id, exact=True),
                )
                return rendered if entry_tokens <= max_tokens else _BUDGET_NOTICE
        elif bucket_type == "letter" and in_scope:
            rendered, entry_tokens = render_stored_bucket(
                exact_bucket,
                _letter_footer(exact_bucket, 0, exact=True),
            )
            return rendered if entry_tokens <= max_tokens else _BUDGET_NOTICE
        elif (
            bucket_type not in ("feel", "plan")
            and _can_surface_search(exact_bucket)
            and in_scope
        ):
            rendered, entry_tokens = render_stored_bucket(
                exact_bucket,
                f"[exact_bucket_id:true] [bucket_id:{exact_bucket['id']}]",
            )
            if entry_tokens > max_tokens:
                return _BUDGET_NOTICE
            if should_touch(str(exact_bucket["id"]), hours=recent_hours):
                asyncio.create_task(
                    rt.bucket_mgr.touch_many([exact_bucket["id"]], ripple=False)
                )
            if rt.fire_webhook:
                await rt.fire_webhook(
                    "breath",
                    {"mode": "exact_id", "matches": 1, "chars": len(rendered)},
                )
            return rendered

    vector_scores, semantic_notice = await _semantic_scores(
        query, top_k=max(max_results, _VECTOR_QUERY_TOPK)
    )
    semantic_diag = _semantic_diagnostics(query, vector_scores, semantic_notice)
    rt.logger.info("op=breath_search phase=semantic diagnostics=%s", semantic_diag)

    search_kwargs = {
        "limit": max(max_results * 4, 20),
        "domain_filter": domain_filter,
        "query_valence": q_valence,
        "query_arousal": q_arousal,
        "vector_scores": vector_scores,
    }
    try:
        raw_matches = await _manager_search(
            query, include_archive=False, **search_kwargs
        )
        all_active = await rt.bucket_mgr.list_all(include_archive=False)
    except Exception as exc:
        rt.logger.error(f"Search failed / 检索失败: {exc}")
        return "检索过程出错，请稍后重试。"

    ordinary_matches = [
        bucket for bucket in raw_matches
        if not _is_archived(bucket)
        and _can_surface_search(bucket)
        and str((bucket.get("metadata", {}) or {}).get("type") or "").lower()
        not in ("feel", "plan", "letter", "archived")
        and _bucket_has_tags(bucket.get("metadata", {}) or {}, tag_filter)
        and _domain_matches(bucket.get("metadata", {}) or {}, domain_filter)
        and _bucket_in_created_range(bucket, created_from, created_to)
    ]
    ordinary_by_id = {
        str(bucket.get("id") or ""): _normal_candidate(bucket)
        for bucket in ordinary_matches
        if bucket.get("id")
    }
    for candidate in ordinary_by_id.values():
        if not candidate["match"] and _literal_hit(query, candidate["bucket"]):
            candidate["direct"] = True
            candidate["match"] = {"literal": True, "semantic": 0.0}

    # Uncapped literal pass over Markdown truth prevents mixed-retriever limits
    # from dropping an explicit original-term hit.
    for bucket in all_active:
        meta = bucket.get("metadata", {}) or {}
        bucket_id = str(bucket.get("id") or "")
        if (
            not bucket_id
            or _is_archived(bucket)
            or not _can_surface_search(bucket)
            or str(meta.get("type") or "").lower()
            in ("feel", "plan", "letter", "archived")
            or not _domain_matches(meta, domain_filter)
            or not _bucket_has_tags(meta, tag_filter)
            or not _bucket_in_created_range(bucket, created_from, created_to)
            or not _literal_hit(query, bucket)
        ):
            continue
        candidate = ordinary_by_id.get(bucket_id) or _normal_candidate(bucket)
        candidate["direct"] = True
        candidate["match"] = {
            **candidate["match"],
            "literal": True,
            "direct": True,
        }
        ordinary_by_id[bucket_id] = candidate
    ordinary = list(ordinary_by_id.values())

    letters = [
        bucket for bucket in all_active
        if not _is_archived(bucket)
        and str((bucket.get("metadata", {}) or {}).get("type") or "").lower()
        == "letter"
        and _bucket_has_tags(bucket.get("metadata", {}) or {}, tag_filter)
        and _domain_matches(bucket.get("metadata", {}) or {}, domain_filter)
        and _bucket_in_created_range(bucket, created_from, created_to)
    ]
    letter_hits = await _letter_candidates(
        query, letters, max(max_results * 2, 8)
    )
    candidates = ordinary + letter_hits

    # Archive is a fallback, never filler. Loading Footprint here is allowed
    # only to exclude archived feel/plan/letter records; it is never rendered.
    if not candidates:
        try:
            archive_matches = await _manager_search(
                query, include_archive=True, **search_kwargs
            )
        except Exception as exc:
            rt.logger.warning(f"archive fallback failed: {exc}")
            archive_matches = []
        for bucket in archive_matches:
            meta = bucket.get("metadata", {}) or {}
            if not _is_archived(bucket):
                continue
            if _archived_original_kind(bucket) in ("feel", "plan", "letter"):
                continue
            if (
                _bucket_has_tags(meta, tag_filter)
                and _domain_matches(meta, domain_filter)
                and _bucket_in_created_range(bucket, created_from, created_to)
            ):
                candidates.append(_normal_candidate(bucket, archived=True))

    # Recent breath results only demote loose ordinary associations. Literal,
    # exact-id and strong semantic results retain their actual rank.
    fresh: list[dict] = []
    inhibited: list[dict] = []
    for candidate in sorted(candidates, key=_candidate_rank, reverse=True):
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
                "search diagnostic: suppressed_duplicate kind=%s id=%s",
                candidate["kind"],
                candidate["bucket"].get("id"),
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
            pseudo = {
                "id": bucket_id,
                "metadata": {},
                "content": candidate["content"],
            }
            rendered, entry_tokens = render_stored_bucket(
                pseudo,
                _letter_footer(bucket, candidate["chunk_index"]),
            )
        else:
            meta = bucket.get("metadata", {}) or {}
            if candidate["kind"] == "archive":
                footer = _archive_footer(bucket_id)
            elif (
                meta.get("pinned")
                or meta.get("protected")
                or meta.get("type") == "permanent"
            ):
                footer = f"📌 [核心准则] [bucket_id:{bucket_id}]"
            elif candidate["match"].get("semantic"):
                footer = f"[语义命中] [bucket_id:{bucket_id}]"
            else:
                footer = f"[bucket_id:{bucket_id}]"
            rendered, entry_tokens = render_stored_bucket(bucket, footer)
        if token_used + entry_tokens > max_tokens:
            budget_blocked = True
            break
        results.append(rendered)
        token_used += entry_tokens
        if (
            candidate["kind"] == "bucket"
            and candidate["direct"]
            and should_touch(bucket_id, hours=recent_hours)
        ):
            touched_ids.append(bucket_id)
        rt.logger.info(
            "search diagnostic: selected kind=%s id=%s score=%.2f direct=%s",
            candidate["kind"],
            bucket_id,
            candidate["score"],
            candidate["direct"],
        )

    if touched_ids:
        asyncio.create_task(rt.bucket_mgr.touch_many(touched_ids, ripple=False))

    if not results:
        if budget_blocked:
            return _BUDGET_NOTICE
        if rt.fire_webhook:
            await rt.fire_webhook("breath", {"mode": "empty", "matches": 0})
        empty_text = (
            f"没有匹配到「{query}」相关的记忆。\n"
            "可以换个更具体的人名、原话或事件关键词再用 breath_search；"
            "它会同时查普通记忆、续接信段落和沉底旧记忆。"
        )
        return (
            f"{empty_text}\n{semantic_notice}" if semantic_notice else empty_text
        )

    final_text = "\n---\n".join(results)
    notices: list[str] = []
    if semantic_notice:
        notices.append(semantic_notice)
    if budget_blocked:
        notices.append(_BUDGET_NOTICE)
    if notices:
        final_text = "\n".join([final_text, *notices])
    if rt.fire_webhook:
        await rt.fire_webhook(
            "breath",
            {"mode": "ok", "matches": len(results), "chars": len(final_text)},
        )
    return final_text
