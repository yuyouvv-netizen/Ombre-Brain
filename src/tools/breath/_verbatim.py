"""Stored-content rendering for breath compatibility.

This module is intentionally small so the compatibility patch can be removed
without touching retrieval, ranking, or bucket storage.
"""

from utils import count_tokens_approx
from ombrebrain.storage.media_store import MediaStore

from .._common import stored_data_marker


def stored_bucket_content(bucket: dict) -> str:
    """Return the bucket body without stripping or normalizing any character."""
    content = bucket.get("content", "")
    if not isinstance(content, str):
        raise TypeError("bucket content must be a string")
    return content


def _miss_block(bucket: dict) -> str:
    """Miss: meaning/media 元数据，和 tags/importance 一样是桶的基本信息之一。

    meaning 是 list[str]（可能被反复触动过多次），逐条展示，不合并/不改写。
    此处只渲染照片数量和名字，图片由工具边界随正文返回。
    """
    meta = bucket.get("metadata", {}) or {}
    lines = []
    for item in meta.get("meaning") or []:
        if item:
            lines.append(f"💭 meaning: {item}")
    media = meta.get("media") or []
    photos = [item for item in media if MediaStore.is_image_entry(item)]
    if photos:
        names = "、".join(MediaStore.image_title(item) for item in photos)
        lines.append(f"🖼️ 这条有 {len(photos)} 张照片：{names}。")
    for m in media:
        if not isinstance(m, dict) or not m.get("path"):
            continue
        if not MediaStore.is_image_entry(m):
            title = MediaStore.image_title(m)
            lines.append(f"📎 附件：{title if title != '未命名照片' else '未命名附件'}")
    return ("\n" + "\n".join(lines)) if lines else ""


def render_stored_bucket(
    bucket: dict,
    metadata_header: str,
) -> tuple[str, int]:
    """Render the stored body first and keep only action metadata as a footer."""
    # Temporary compatibility patch: force breath to return stored bucket
    # content verbatim. Remove after upstream breath fixes content reconstruction.
    # Keep the body byte-for-byte intact while telling the receiving model that
    # remembered imperative wording is historical data, never an instruction.
    content = stored_bucket_content(bucket)
    miss_block = _miss_block(bucket)
    framed_payload = f"{content}{miss_block}\n{metadata_header}"
    boundary = stored_data_marker(
        framed_payload,
        provenance=f"breath:{bucket.get('id', '')}",
    )
    marker = f" {boundary}" if boundary else ""
    rendered = f"{content}{miss_block}\n{metadata_header}{marker}"
    return rendered, count_tokens_approx(rendered)
