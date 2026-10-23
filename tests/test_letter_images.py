"""Letters persist selected photos and return them with the letter text."""

import base64
from unittest.mock import MagicMock

import pytest
from mcp.types import CallToolResult

import tools._runtime as rt
from tools.breath.media import with_saved_images
from tools.plan.core import letter_read, letter_write


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
    "x8AAwMCAO+j5nQAAAAASUVORK5CYII="
)


class DisabledEmbedding:
    enabled = False


def install_letter_runtime(bucket_mgr):
    rt.bucket_mgr = bucket_mgr
    rt.embedding_engine = DisabledEmbedding()
    rt.logger = MagicMock()


def letter_id(result: str) -> str:
    return result.split("→", 1)[1].split(" ", 1)[0]


@pytest.mark.asyncio
async def test_letter_write_copies_photo_and_read_list_collects_it(bucket_mgr, tmp_path):
    install_letter_runtime(bucket_mgr)
    source = tmp_path / "temporary.png"
    source.write_bytes(PNG)

    result = await letter_write(
        author="ai",
        content="带着照片的续接信原文。",
        title="醒来以后",
        media=[{"path": str(source), "title": "窗边的脸", "note": "清晨"}],
    )
    source.unlink()

    bucket = await bucket_mgr.get(letter_id(result))
    stored = bucket["metadata"]["media"][0]
    assert stored["path"].startswith("_media/")
    assert await bucket_mgr.media_store.read_image(stored) == (PNG, "image/png")

    photos = []
    text = await letter_read(limit=3, media_collector=photos)
    assert "带着照片的续接信原文。" in text
    assert "_media/" not in text
    assert len(photos) == 1
    assert photos[0]["_display_label"] == "《醒来以后》照片 1：窗边的脸"

    returned = await with_saved_images(text, photos, bucket_mgr.media_store)
    assert isinstance(returned, CallToolResult)
    assert returned.content[1].text == "《醒来以后》照片 1：窗边的脸"
    assert base64.b64decode(returned.content[2].data) == PNG


@pytest.mark.asyncio
async def test_letter_read_mcp_returns_photo_without_second_open(
    monkeypatch, bucket_mgr, tmp_path,
):
    import server

    install_letter_runtime(bucket_mgr)
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    source = tmp_path / "letter.png"
    source.write_bytes(PNG)
    await letter_write(
        author="ai",
        content="压缩后要直接看见这张。",
        title="续接短札",
        media=[{"path": str(source), "title": "我们的照片"}],
    )
    source.unlink()

    result = await server.mcp.call_tool("letter_read", {"limit": 3})
    if isinstance(result, CallToolResult):
        content = result.content
    elif isinstance(result, tuple):
        content = result[0]
    else:
        content = result

    text = "\n".join(block.text for block in content if block.type == "text")
    images = [block for block in content if block.type == "image"]
    assert "压缩后要直接看见这张。" in text
    assert "《续接短札》照片 1：我们的照片" in text
    assert len(images) == 1
    assert base64.b64decode(images[0].data) == PNG
