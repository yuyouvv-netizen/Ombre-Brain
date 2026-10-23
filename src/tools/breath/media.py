"""只把本次实际返回正文的记忆照片附为原生图片块。"""

import base64

from mcp.types import CallToolResult, ImageContent, TextContent
from ombrebrain.storage.media_store import MediaStore


def collect_saved_images(bucket: dict, collector: list | None) -> None:
    """正文通过过滤与预算后收集照片；保留来源，不修改存储元数据。"""
    if collector is None:
        return
    meta = bucket.get("metadata") or {}
    bucket_id = str(bucket.get("id") or "")
    seen = {(item.get("_bucket_id"), item.get("path")) for item in collector if isinstance(item, dict)}
    for entry in meta.get("media") or []:
        if not MediaStore.is_image_entry(entry):
            continue
        key = (bucket_id, entry["path"])
        if key in seen:
            continue
        seen.add(key)
        collector.append({
            **entry,
            "_bucket_id": bucket_id,
            "_display_label": f"[bucket_id:{bucket_id}] 照片：{MediaStore.image_title(entry)}",
        })


async def with_saved_images(text: str, entries: list, store):
    if not entries:
        return text
    content = [TextContent(type="text", text=text)]
    total = 0
    for index, entry in enumerate(entries[:6], start=1):
        try:
            if not isinstance(entry, dict):
                raise ValueError("invalid media metadata")
            data, mime = await store.read_image(entry)
            total += len(data)
            if total > 24 * 1024 * 1024:
                raise ValueError("response limit")
            label = " ".join(str(entry.get("_display_label") or "").split())[:300]
            if not label:
                title = store.image_title(entry)
                label = f"已存图片 {index}：{title}"
            content.append(TextContent(type="text", text=label))
            content.append(ImageContent(type="image", mimeType=mime, data=base64.b64encode(data).decode("ascii")))
        except (OSError, ValueError, TypeError):
            # Never expose local paths, file contents or base64 in error messages.
            content.append(TextContent(type="text", text=f"已存图片 {index} 暂时无法读取或不是支持的图片；文字记忆仍可正常阅读。"))
    if len(entries) > 6:
        content.append(TextContent(type="text", text="本次只显示前 6 张图片，其余引用仍保留在记忆中。"))
    return CallToolResult(content=content)
