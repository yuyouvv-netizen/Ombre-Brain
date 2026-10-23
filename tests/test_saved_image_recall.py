"""Selected photos survive source expiry and return through the existing MCP."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from mcp.types import CallToolResult

from ombrebrain.storage.media_store import MediaStore
from tools.breath.media import with_saved_images
from tools.breath._verbatim import render_stored_bucket


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j5nQAAAAASUVORK5CYII=")


def test_photo_hint_keeps_body_and_id_without_file_paths():
    bucket = {"id": "photo-memory", "content": "原话\n不能改写。", "metadata": {"media": [
        {"path": "_media/photo-memory/one.png", "title": "老婆的脸"},
    ]}}
    rendered, _ = render_stored_bucket(bucket, "[bucket_id:photo-memory]")
    assert rendered.startswith(bucket["content"] + "\n")
    assert "这条有 1 张照片：老婆的脸，用编号打开可以看。" in rendered
    assert "[bucket_id:photo-memory]" in rendered
    assert "_media/" not in rendered
    bucket["metadata"]["media"].extend([
        {"path": "_media/photo-memory/two.jpg"},
        {"path": "_media/photo-memory/voice.ogg", "title": "声音"},
        {"path": None},
    ])
    rendered, _ = render_stored_bucket(bucket, "[bucket_id:photo-memory]")
    assert "这条有 2 张照片：老婆的脸、未命名照片" in rendered
    assert "附件：声音" in rendered
    assert "_media/" not in rendered


def _photo_web_app(monkeypatch, manager):
    import time
    from starlette.applications import Starlette
    from starlette.routing import Route
    from web import buckets

    class Routes:
        def __init__(self):
            self.routes = []

        def custom_route(self, path, methods):
            def register(handler):
                self.routes.append(Route(path, handler, methods=methods))
                return handler
            return register

    monkeypatch.setattr(buckets.sh, "bucket_mgr", manager)
    monkeypatch.setattr(buckets.sh, "decay_engine", SimpleNamespace(calculate_score=lambda _: 1.0))
    monkeypatch.setattr(buckets.sh, "_is_setup_needed", lambda: False)
    monkeypatch.setattr(buckets.sh, "_sessions", {"photo-test-session": time.time() + 60})
    routes = Routes()
    buckets.register(routes)
    return Starlette(routes=routes.routes)


@pytest.mark.asyncio
async def test_dashboard_photos_require_session_and_preserve_original(monkeypatch, bucket_mgr):
    import httpx
    from web import buckets

    bucket_id = await bucket_mgr.create(content="测试照片原文", media=[
        {"data_base64": base64.b64encode(b"audio").decode(), "filename": "voice.ogg"},
        {"data_base64": base64.b64encode(PNG).decode(), "filename": "face.png", "title": "老婆的脸", "note": "一句话"},
    ])
    app = _photo_web_app(monkeypatch, bucket_mgr)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        url = f"/api/bucket/{bucket_id}/photo/1"
        assert (await client.get(url)).status_code == 401
        client.cookies.set("ombre_session", "photo-test-session")
        detail = (await client.get(f"/api/bucket/{bucket_id}")).json()
        assert detail["content"] == "测试照片原文"
        assert detail["photos"] == [{"index": 1, "title": "老婆的脸", "note": "一句话"}]
        response = await client.get(url)
        assert response.status_code == 200
        assert response.content == PNG
        assert response.headers["content-type"] == "image/png"
        assert response.headers["cache-control"] == "private, no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        for suffix in ("0", "2", "-1", "not-an-index"):
            assert (await client.get(f"/api/bucket/{bucket_id}/photo/{suffix}")).status_code == 404
        assert (await client.get("/api/bucket/missing/photo/0")).status_code == 404
        buckets.sh._sessions.clear()
        assert (await client.get(url)).status_code == 401


@pytest.mark.asyncio
async def test_dashboard_rejects_unsafe_and_missing_photos(monkeypatch, tmp_path):
    import httpx

    store = MediaStore(str(tmp_path), str(tmp_path / "_media"))
    entries = await store.persist("fixture", [{"data_base64": base64.b64encode(PNG).decode(), "filename": "photo.png"}])
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG)
    (store.media_dir / "link.png").symlink_to(outside)
    unsafe = [
        {"path": str(outside)}, {"path": "_media/../outside.png"},
        {"path": "_media/link.png"}, {"path": "_media/missing.png"},
        {**entries[0], "sha256": "0" * 64},
    ]
    fake = await store.persist("fake", [{"data_base64": base64.b64encode(b"secret-not-image").decode(), "filename": "fake.png"}])
    unsafe.extend(fake)
    manager = SimpleNamespace(media_store=store, get=AsyncMock())
    app = _photo_web_app(monkeypatch, manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", cookies={"ombre_session": "photo-test-session"}) as client:
        for entry in unsafe:
            manager.get.return_value = {"id": "fixture", "metadata": {"media": [entry]}}
            response = await client.get("/api/bucket/fixture/photo/0")
            assert response.status_code == 404
            assert str(tmp_path) not in response.text
            assert "secret-not-image" not in response.text


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
