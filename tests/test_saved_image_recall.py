"""Selected photos survive source expiry and return through the existing MCP."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from mcp.types import CallToolResult

from ombrebrain.storage.media_store import MediaStore
from tools.breath.media import with_saved_images


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j5nQAAAAASUVORK5CYII=")


@pytest.mark.asyncio
async def test_upload_expire_reopen_and_replace(tmp_path):
    vault = tmp_path / "vault"
    store = MediaStore(str(vault), str(vault / "_media"))
    source = tmp_path / "temp.png"
    source.write_bytes(PNG)
    entries = await store.persist("fixture", [{"data_base64": base64.b64encode(source.read_bytes()).decode(), "filename": "photo.png", "type": "image/png"}])
    source.unlink()
    result = await with_saved_images("verbatim memory", entries, store)
    assert isinstance(result, CallToolResult)
    assert result.content[0].text == "verbatim memory"
    assert base64.b64decode(result.content[2].data) == PNG
    assert result.content[2].mimeType == "image/png"
    replaced = await store.persist("fixture", entries)
    assert replaced[0]["path"] == entries[0]["path"]
    assert await store.read_image(replaced[0]) == (PNG, "image/png")


@pytest.mark.asyncio
async def test_stored_read_rejects_escape_symlinks_and_spoofed_types(tmp_path):
    store = MediaStore(str(tmp_path), str(tmp_path / "_media"))
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG)
    (store.media_dir / "escape.png").symlink_to(outside)
    (store.media_dir / "directory").symlink_to(tmp_path, target_is_directory=True)
    for path in [str(outside), "_media/../outside.png", "_media/escape.png", "_media/directory/outside.png"]:
        with pytest.raises((ValueError, OSError)):
            await store.read_image({"path": path})
    entries = await store.persist("fake", [{"data_base64": base64.b64encode(b"secret text").decode(), "filename": "photo.png", "type": "image/png"}])
    result = await with_saved_images("text survives", entries, store)
    assert all(block.type == "text" for block in result.content)
    assert "secret text" not in str(result)


@pytest.mark.asyncio
async def test_hash_mismatch_missing_and_limits_preserve_text(tmp_path):
    store = MediaStore(str(tmp_path), str(tmp_path / "_media"))
    entries = await store.persist("fixture", [{"data_base64": base64.b64encode(PNG).decode(), "filename": "photo.png"}])
    with pytest.raises(ValueError):
        await store.read_image({**entries[0], "sha256": "0" * 64})
    result = await with_saved_images("plain", entries * 7, store)
    assert sum(x.type == "image" for x in result.content) == 6
    (tmp_path / entries[0]["path"]).unlink()
    result = await with_saved_images("plain", entries, store)
    assert result.content[0].text == "plain"
    assert not any(x.type == "image" for x in result.content)
    assert await with_saved_images("plain", [], store) == "plain"


@pytest.mark.asyncio
async def test_only_in_scope_exact_id_collects_images(monkeypatch, bucket_mgr):
    import tools._runtime as rt
    from tools.breath import dispatch
    monkeypatch.setattr(rt, "config", {"surfacing": {}})
    monkeypatch.setattr(rt, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(rt, "decay_engine", SimpleNamespace(ensure_started=AsyncMock()))
    monkeypatch.setattr(rt, "embedding_engine", SimpleNamespace(enabled=False))
    monkeypatch.setattr(rt, "logger", MagicMock())
    monkeypatch.setattr(rt, "fire_webhook", None)
    monkeypatch.setattr(rt, "mark_op", None)
    monkeypatch.setattr(rt, "record_v3_tool_event", lambda *_args, **_kwargs: None)
    bucket_id = await bucket_mgr.create(content="selected photo fixture", domain=["fixture"], media=[{"data_base64": base64.b64encode(PNG).decode(), "filename": "photo.png"}])
    entries = []
    text = await dispatch(query=bucket_id, media_collector=entries)
    assert "selected photo fixture" in text
    assert len(entries) == 1
    for args in [{"query": "selected photo"}, {"query": bucket_id, "domain": "other"}, {"query": bucket_id, "date_from": "2099-01-01"}, {"query": bucket_id, "max_tokens": 1}, {"query": bucket_id, "catalog": True}]:
        entries = []
        await dispatch(**args, media_collector=entries)
        assert entries == []


@pytest.mark.asyncio
async def test_mcp_boundary_keeps_native_image_blocks(monkeypatch, bucket_mgr):
    import server
    entries = await bucket_mgr.media_store.persist("fixture", [{"data_base64": base64.b64encode(PNG).decode(), "filename": "photo.png"}])

    async def dispatch(**kwargs):
        kwargs["media_collector"].extend(entries)
        return "verbatim result"

    monkeypatch.setattr(server._t_breath, "dispatch", dispatch)
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    result = await server.mcp.call_tool("breath_search", {"query": "fixture"})
    # FastMCP versions may return CallToolResult directly or a content tuple.
    if isinstance(result, CallToolResult):
        content = result.content
    elif isinstance(result, tuple):
        content = result[0]
    else:
        content = result
    assert content[0].type == "text"
    images = [item for item in content if item.type == "image"]
    assert len(images) == 1
    assert base64.b64decode(images[0].data) == PNG
