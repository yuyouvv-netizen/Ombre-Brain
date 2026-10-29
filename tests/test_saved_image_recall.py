"""Selected photos survive source expiry and return through the existing MCP."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from mcp.types import CallToolResult

from ombrebrain.storage.media_store import MediaStore
from tools.breath.media import collect_saved_images, with_saved_images
from tools.breath._verbatim import render_stored_bucket


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j5nQAAAAASUVORK5CYII=")


def test_photo_hint_keeps_body_and_id_without_file_paths():
    bucket = {"id": "photo-memory", "content": "原话\n不能改写。", "metadata": {"media": [
        {"path": "_media/photo-memory/one.png", "title": "老婆的脸"},
    ]}}
    rendered, _ = render_stored_bucket(bucket, "[bucket_id:photo-memory]")
    assert rendered.startswith(bucket["content"] + "\n")
    assert "这条有 1 张照片：老婆的脸。" in rendered
    assert "用编号打开" not in rendered
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


@pytest.mark.asyncio
@pytest.mark.parametrize("bucket_type", ["dynamic", "letter"])
async def test_dashboard_photo_edit_roundtrip(monkeypatch, bucket_mgr, bucket_type):
    import httpx
    from pathlib import Path

    bid = await bucket_mgr.create(content="照片测试 [[原文]]", bucket_type=bucket_type, media=[
        {"data_base64": base64.b64encode(b"voice").decode(), "filename": "voice.ogg"},
    ])
    original = await bucket_mgr.get(bid)
    app = _photo_web_app(monkeypatch, bucket_mgr)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        url = f"/api/bucket/{bid}/photos"
        assert (await client.put(url, json={"photos": []})).status_code == 401
        client.cookies.set("ombre_session", "photo-test-session")
        detail = (await client.get(f"/api/bucket/{bid}")).json()
        payload = {"revision": detail["photos_revision"], "photos": [{
            "data_base64": base64.b64encode(PNG).decode(), "title": "合照", "note": "第一次",
        }]}
        assert (await client.put(url, json=payload)).status_code == 200
        assert (await client.put(url, json=payload)).status_code == 409
        detail = (await client.get(f"/api/bucket/{bid}")).json()
        assert detail["photos"] == [{"index": 1, "title": "合照", "note": "第一次"}]
        assert (await client.get(f"/api/bucket/{bid}/photo/1")).content == PNG
        media = detail["metadata"]["media"]
        stored = bucket_mgr.media_store.vault_dir / media[1]["path"]
        payload = {"revision": detail["photos_revision"], "photos": [{"index": 1, "title": "改名", "note": ""}]}
        assert (await client.put(url, json=payload)).status_code == 200
        updated = await bucket_mgr.get(bid)
        assert updated["metadata"]["media"][0] == media[0]
        assert updated["metadata"]["media"][1]["title"] == "改名"
        assert not updated["metadata"]["media"][1].get("note")
        assert updated["content"] == original["content"]
        assert updated["metadata"]["last_active"] == original["metadata"]["last_active"]
        assert updated["metadata"]["activation_count"] == original["metadata"]["activation_count"]
        detail = (await client.get(f"/api/bucket/{bid}")).json()
        assert (await client.put(url, json={"revision": detail["photos_revision"], "photos": []})).status_code == 200
        assert Path(stored).read_bytes() == PNG
        assert (await bucket_mgr.get(bid))["metadata"]["media"] == [media[0]]


@pytest.mark.asyncio
async def test_dashboard_photo_edit_rejects_invalid_inputs_and_concurrent_append(monkeypatch, bucket_mgr):
    import httpx

    bid = await bucket_mgr.create(content="不能被覆盖的原文")
    app = _photo_web_app(monkeypatch, bucket_mgr)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", cookies={"ombre_session": "photo-test-session"}) as client:
        detail = (await client.get(f"/api/bucket/{bid}")).json()
        valid = {"data_base64": base64.b64encode(PNG).decode()}
        for photos in [None, {}, ["photo"], [{"path": "/etc/passwd"}], [{"index": True}], [{"index": 0}],
                       [dict(valid, title=42)], [dict(valid, note="x" * 501)], [dict(valid, data_base64="!bad")],
                       [dict(valid, data_base64=base64.b64encode(b"not-image").decode())], [valid] * 7]:
            response = await client.put(f"/api/bucket/{bid}/photos", json={"revision": detail["photos_revision"], "photos": photos})
            assert response.status_code == 400
            assert not (await bucket_mgr.get(bid))["metadata"].get("media")
        update = bucket_mgr.update

        async def append_before_commit(bucket_id, **kwargs):
            await update(bucket_id, media_append=[valid])
            return await update(bucket_id, **kwargs)

        monkeypatch.setattr(bucket_mgr, "update", append_before_commit)
        response = await client.put(f"/api/bucket/{bid}/photos", json={"revision": detail["photos_revision"], "photos": []})
        assert response.status_code == 409
        assert len((await bucket_mgr.get(bid))["metadata"]["media"]) == 1
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
async def test_only_returned_in_scope_memories_collect_images(monkeypatch, bucket_mgr):
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
    entries = []
    text = await dispatch(query="selected photo", media_collector=entries)
    assert "selected photo fixture" in text
    assert len(entries) == 1
    assert bucket_id in entries[0]["_display_label"]
    for args in [{"query": bucket_id, "domain": "other"}, {"query": bucket_id, "date_from": "2099-01-01"}, {"query": bucket_id, "max_tokens": 1}, {"query": bucket_id, "catalog": True}, {"query": "selected photo", "max_tokens": 1}]:
        entries = []
        await dispatch(**args, media_collector=entries)
        assert entries == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,args", [
    ("breath", {}), ("breath_search", {"query": "fixture"}),
    ("breath_advanced", {"query": "fixture"}), ("dream", {}),
])
async def test_mcp_boundary_keeps_native_image_blocks(monkeypatch, bucket_mgr, tool, args):
    import server
    entries = await bucket_mgr.media_store.persist("fixture", [{"data_base64": base64.b64encode(PNG).decode(), "filename": "photo.png"}])

    async def dispatch(**kwargs):
        kwargs["media_collector"].extend(entries)
        return "verbatim result"

    monkeypatch.setattr(server._t_breath, "dispatch", dispatch)
    monkeypatch.setattr(server._t_dream, "dispatch", dispatch)
    monkeypatch.setattr(server, "bucket_mgr", bucket_mgr)
    result = await server.mcp.call_tool(tool, args)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("args,kind", [
    ({}, "dynamic"),
    ({}, "permanent"),
    ({"importance_min": 5}, "dynamic"),
    ({"domain": "feel"}, "feel"),
])
async def test_surface_modes_attach_only_visible_budgeted_photos(monkeypatch, bucket_mgr, args, kind):
    import tools._runtime as rt
    from tools.breath import dispatch

    monkeypatch.setattr(rt, "config", {"surfacing": {}})
    monkeypatch.setattr(rt, "bucket_mgr", bucket_mgr)
    monkeypatch.setattr(rt, "decay_engine", SimpleNamespace(
        ensure_started=AsyncMock(), calculate_score=lambda _: 10,
    ))
    monkeypatch.setattr(rt, "logger", MagicMock())
    monkeypatch.setattr(rt, "fire_webhook", None)
    monkeypatch.setattr(rt, "mark_op", None)
    monkeypatch.setattr(rt, "record_v3_tool_event", lambda *_args, **_kwargs: None)
    bucket_id = await bucket_mgr.create(
        content="今天留下的照片和原文", bucket_type=kind,
        media=[{"data_base64": base64.b64encode(PNG).decode(), "filename": "photo.png"}],
    )
    media = []
    text = await dispatch(**args, media_collector=media)
    assert "今天留下的照片和原文" in text
    assert len(media) == 1
    assert media[0]["_bucket_id"] == bucket_id
    result = await with_saved_images(text, media, bucket_mgr.media_store)
    assert any(block.type == "image" for block in result.content)
    for extra in [{"max_tokens": 1}, {"catalog": True}]:
        media = []
        await dispatch(**args, **extra, media_collector=media)
        assert media == []


def test_collector_deduplicates_and_does_not_mutate_metadata():
    entry = {"path": "_media/example/one.png", "title": "同一张照片"}
    bucket = {"id": "example", "metadata": {"media": [
        entry, entry.copy(), {"path": "_media/example/audio.ogg"},
    ]}}
    media = []
    collect_saved_images(bucket, media)
    collect_saved_images(bucket, media)
    assert len(media) == 1
    assert "_bucket_id" not in entry
    assert "example" in media[0]["_display_label"]


def test_dream_photos_follow_rendered_bodies_and_budget(monkeypatch):
    import tools._runtime as rt
    from tools.dream.output import format_dream_output

    monkeypatch.setattr(rt, "config", {"surfacing": {"dream_max_tokens": 1000}})
    monkeypatch.setattr(rt, "logger", MagicMock())
    def bucket(bid, content):
        return {"id": bid, "content": content, "metadata": {
            "type": "dynamic", "media": [{"path": f"_media/{bid}/one.png"}],
        }}
    shown = bucket("shown", "应该一起看到的原文和照片")
    omitted = bucket("omitted", "太长的记忆" * 5000)
    core = bucket("core", "核心记忆")
    media = []
    text = format_dream_output(
        recent=[shown, omitted], all_buckets=[], window_hours=48,
        connection_hint="", crystal_hint="", core_context=[core],
        media_collector=media,
    )
    assert shown["content"] in text
    assert omitted["content"] not in text
    assert {entry["_bucket_id"] for entry in media} == {"shown", "core"}
    assert "_media/" not in text
