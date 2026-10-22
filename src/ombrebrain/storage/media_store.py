"""OB 媒体持久化存储。

本模块把 MCP 调用携带的服务器可读临时文件或 Base64 数据复制到持久媒体目录，
并返回可写入 Markdown frontmatter 的稳定元数据。它不理解记忆内容、不操作桶文件，
也不会因为记忆归档而删除媒体。对外暴露 ``MediaStore`` 和
``MediaPersistenceError``。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import mimetypes
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

_SAFE_SUFFIX = re.compile(r"^\.[a-zA-Z0-9]{1,10}$")
_DEFAULT_MAX_MEDIA_BYTES = 25 * 1024 * 1024


class MediaPersistenceError(ValueError):
    """媒体无法在 OB 服务器上永久保存。"""


class MediaStore:
    """把媒体复制到持久目录，并生成稳定引用。"""

    def __init__(
        self,
        vault_dir: str,
        media_dir: str,
        *,
        max_bytes: int = _DEFAULT_MAX_MEDIA_BYTES,
    ) -> None:
        self.vault_dir = Path(vault_dir).resolve()
        self.media_dir = Path(media_dir).resolve()
        self.max_bytes = max(1, int(max_bytes))
        self.media_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _suffix(name: str, mime_type: str) -> str:
        suffix = Path(name).suffix.lower()
        if _SAFE_SUFFIX.fullmatch(suffix):
            return suffix
        guessed = mimetypes.guess_extension(mime_type or "") or ".bin"
        return guessed if _SAFE_SUFFIX.fullmatch(guessed) else ".bin"

    def _stable_path(self, bucket_id: str, digest: str, suffix: str) -> Path:
        safe_bucket = re.sub(r"[^a-zA-Z0-9_.-]", "_", bucket_id)[:128]
        target_dir = (self.media_dir / safe_bucket).resolve()
        if self.media_dir not in target_dir.parents:
            raise MediaPersistenceError("媒体目录越界，已拒绝保存。")
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{digest}{suffix}"

    def _frontmatter_path(self, target: Path) -> str:
        try:
            return target.relative_to(self.vault_dir).as_posix()
        except ValueError:
            return str(target)

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        """在目标目录内写临时文件后原子替换，避免崩溃留下半张媒体。"""
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def _read_path(self, raw_path: str) -> tuple[bytes, str]:
        source = Path(raw_path).expanduser()
        try:
            before_open = os.lstat(source)
        except OSError as exc:
            raise MediaPersistenceError(
                f"媒体临时路径在 OB 服务器上不可读：{raw_path}。"
                "请改传 data_base64，不能把客户端临时路径直接写进记忆。"
            ) from exc
        if stat.S_ISLNK(before_open.st_mode):
            raise MediaPersistenceError(
                f"媒体路径必须是普通文件，不能是符号链接：{raw_path}"
            )
        if not stat.S_ISREG(before_open.st_mode):
            raise MediaPersistenceError(
                f"媒体路径必须是普通文件：{raw_path}"
            )

        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd: int | None = None
        try:
            fd = os.open(source, flags)
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise MediaPersistenceError(
                    f"媒体路径必须是普通文件：{raw_path}"
                )

            # 以已打开的文件描述符为读取真源。打开后再比较路径身份，
            # 可在不二次按路径读取的前提下检出并发替换。
            after_open = os.lstat(source)
            if stat.S_ISLNK(after_open.st_mode) or (
                after_open.st_dev,
                after_open.st_ino,
            ) != (opened.st_dev, opened.st_ino):
                raise MediaPersistenceError(
                    f"媒体路径在打开期间发生变化：{raw_path}"
                )
            if opened.st_size > self.max_bytes:
                raise MediaPersistenceError(
                    f"媒体文件超过单项上限 {self.max_bytes} 字节：{raw_path}"
                )

            with os.fdopen(fd, "rb") as handle:
                fd = None
                data = handle.read(self.max_bytes + 1)
            if len(data) > self.max_bytes:
                raise MediaPersistenceError(
                    f"媒体文件超过单项上限 {self.max_bytes} 字节：{raw_path}"
                )
            return data, source.name
        except MediaPersistenceError:
            raise
        except OSError as exc:
            raise MediaPersistenceError(
                f"媒体临时路径在 OB 服务器上不可读：{raw_path}。"
                "请改传 data_base64，不能把客户端临时路径直接写进记忆。"
            ) from exc
        finally:
            if fd is not None:
                os.close(fd)

    def _decode_base64(self, value: str) -> bytes:
        payload = value.strip()
        if payload.startswith("data:"):
            _, separator, payload = payload.partition(",")
            if not separator:
                raise MediaPersistenceError("媒体 data URI 缺少数据部分。")
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise MediaPersistenceError("媒体 data_base64 不是有效 Base64。") from exc
        if len(data) > self.max_bytes:
            raise MediaPersistenceError(
                f"媒体数据超过单项上限 {self.max_bytes} 字节。"
            )
        return data

    def _read_stored(self, entry: dict) -> bytes:
        """Read only inside media_dir; openat rejects symlinks at every level."""
        raw = Path(str(entry.get("path") or ""))
        target = raw if raw.is_absolute() else self.vault_dir / raw
        try:
            parts = target.relative_to(self.media_dir).parts
        except ValueError as exc:
            raise MediaPersistenceError("媒体引用不在持久媒体目录中。") from exc
        if not parts or any(part in (".", "..", "") for part in parts):
            raise MediaPersistenceError("媒体引用路径无效。")
        directory = os.open(self.media_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fd = None
        try:
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            info = os.fstat(fd)
            limit = min(self.max_bytes, 12 * 1024 * 1024)
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
                raise MediaPersistenceError("已存图片为空、过大或不是普通文件。")
            with os.fdopen(fd, "rb") as handle:
                fd = None
                data = handle.read(limit + 1)
            if len(data) != info.st_size or len(data) > limit:
                raise MediaPersistenceError("已存图片读取不完整。")
            if entry.get("sha256") and hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise MediaPersistenceError("已存图片校验失败。")
            return data
        finally:
            os.close(directory)
            if fd is not None:
                os.close(fd)

    async def read_image(self, entry: dict) -> tuple[bytes, str]:
        data = await asyncio.to_thread(self._read_stored, entry)
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif data.startswith((b"GIF87a", b"GIF89a")):
            mime = "image/gif"
        elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            mime = "image/webp"
        else:
            raise MediaPersistenceError("这项媒体不是支持的图片。")
        return data, mime

    def _persist_one(self, bucket_id: str, item: Any) -> dict[str, Any]:
        entry = {"path": item} if isinstance(item, str) else dict(item or {})
        mime_type = str(entry.get("type") or entry.get("mime_type") or "")[:128]
        if entry.get("data_base64"):
            data = self._decode_base64(str(entry["data_base64"]))
            source_name = str(entry.get("filename") or entry.get("title") or "media")
        else:
            raw_path = str(entry.get("path") or "").strip()
            if not raw_path:
                raise MediaPersistenceError("media 每项必须提供 path 或 data_base64。")
            # Stable vault-relative references can be retained when replacing media.
            if raw_path.startswith("_media/"):
                data = self._read_stored({"path": raw_path})
                source_name = Path(raw_path).name
            else:
                data, source_name = self._read_path(raw_path)
        digest = hashlib.sha256(data).hexdigest()
        suffix = self._suffix(source_name, mime_type)
        target = self._stable_path(bucket_id, digest, suffix)
        if not target.exists():
            self._atomic_write(target, data)
        result: dict[str, Any] = {
            "path": self._frontmatter_path(target),
            "sha256": digest,
            "size": len(data),
            "stored": True,
        }
        for key, limit in (("title", 200), ("type", 128), ("note", 500)):
            value = entry.get(key)
            if value:
                result[key] = str(value)[:limit]
        return result

    async def persist(self, bucket_id: str, media: Any) -> list[dict[str, Any]]:
        """永久保存一项或多项媒体；任何一项失败则明确报错。"""
        if not media:
            return []
        items = media if isinstance(media, list) else [media]
        return await asyncio.to_thread(
            lambda: [self._persist_one(bucket_id, item) for item in items]
        )
