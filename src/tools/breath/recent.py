"""Short-lived read memory used to avoid echoing the same associations.

This state is deliberately process-local and never changes bucket metadata.  A
normal ``breath()`` records the ordinary memories it actually showed; later
searches may demote those memories when they are only loose associations.  A
literal, exact-id, or strong semantic match always bypasses the inhibition.
"""

from __future__ import annotations

import time


_surfaced_at: dict[str, float] = {}
_touched_at: dict[str, float] = {}


def _prune(store: dict[str, float], window_seconds: float, now: float) -> None:
    expired = [bucket_id for bucket_id, seen_at in store.items() if now - seen_at >= window_seconds]
    for bucket_id in expired:
        store.pop(bucket_id, None)


def record_surfaced(bucket_ids: list[str], *, hours: float = 12.0) -> None:
    """Remember ordinary bucket ids shown by spontaneous breath for ``hours``."""
    now = time.monotonic()
    window = max(0.0, float(hours)) * 3600.0
    _prune(_surfaced_at, window, now)
    for bucket_id in bucket_ids:
        if bucket_id:
            _surfaced_at[str(bucket_id)] = now


def was_recently_surfaced(bucket_id: str, *, hours: float = 12.0) -> bool:
    now = time.monotonic()
    window = max(0.0, float(hours)) * 3600.0
    _prune(_surfaced_at, window, now)
    seen_at = _surfaced_at.get(str(bucket_id))
    return seen_at is not None and now - seen_at < window


def should_touch(bucket_id: str, *, hours: float = 12.0) -> bool:
    """Return true once per window for deliberate, high-confidence retrievals."""
    now = time.monotonic()
    window = max(0.0, float(hours)) * 3600.0
    _prune(_touched_at, window, now)
    key = str(bucket_id)
    seen_at = _touched_at.get(key)
    if seen_at is not None and now - seen_at < window:
        return False
    _touched_at[key] = now
    return True


def clear() -> None:
    """Test helper; production callers should let the TTL expire naturally."""
    _surfaced_at.clear()
    _touched_at.clear()
