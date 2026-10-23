"""Attach saved photos only to explicitly opened, in-scope memory IDs."""

import base64

from mcp.types import CallToolResult, ImageContent, TextContent


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
