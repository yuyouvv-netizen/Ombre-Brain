"""稳定记忆评分与检索索引维护。

普通记忆只按稳定重要度评分，不随时间、已解决或已消化状态降权。
保留 DecayEngine 名称和周期统计字段兼容现有调用；后台只补齐索引，
不自动结案、归档或恢复任何记忆。
"""

import asyncio
import logging
from datetime import datetime

from utils import parse_iso_datetime

logger = logging.getLogger("ombre_brain.decay")

_DEFAULT_DAYS_FALLBACK = 30
_SECONDS_PER_DAY = 86400
_SECONDS_PER_HOUR = 3600
_BACKFILL_MAX_PER_CYCLE = 50

def _days_since_active(meta: dict, fallback_days: float = _DEFAULT_DAYS_FALLBACK) -> float:
    """保留旧时间解析接口供诊断使用，不再参与评分或生命周期决策。"""
    if not isinstance(meta, dict):
        return fallback_days
    raw = meta.get("last_active") or meta.get("created") or ""
    try:
        last_active = parse_iso_datetime(raw)
        return max(0.0, (datetime.now() - last_active).total_seconds() / _SECONDS_PER_DAY)
    except (ValueError, TypeError):
        return float(fallback_days)


class DecayEngine:
    """兼容原接口的稳定评分与索引维护服务。"""

    def __init__(self, config: dict, bucket_mgr):
        decay_cfg = config.get("decay", {})
        # 旧参数只保留读取兼容性，不能重新启用时间降权或自动归档。
        self.decay_lambda = decay_cfg.get("lambda", 0.05)
        self.threshold = decay_cfg.get("threshold", 0.3)
        self.check_interval = decay_cfg.get("check_interval_hours", 24)
        self.bucket_mgr = bucket_mgr
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def _calc_time_weight(days_since: float) -> float:
        """兼容旧调用，时间始终不影响记忆权重。"""
        return 1.0

    def calculate_score(self, metadata: dict) -> float:
        """普通记忆按重要度排序；特殊类型保留原专用通道分值。"""
        if not isinstance(metadata, dict):
            return 0.0
        if metadata.get("pinned") or metadata.get("protected") or metadata.get("type") == "permanent":
            return 999.0
        if metadata.get("type") in ("feel", "plan", "letter"):
            return 50.0
        try:
            return float(max(1, min(10, int(metadata.get("importance", 5)))))
        except (TypeError, ValueError, OverflowError):
            return 5.0

    async def run_decay_cycle(self) -> dict:
        """只维护检索索引；不改原文、状态、激活时间或归档位置。"""
        try:
            buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as exc:
            logger.error("维护周期读取记忆失败: %s", exc)
            return {"checked": 0, "archived": 0, "lowest_score": 0, "error": str(exc)}
        scores = []
        for bucket in buckets:
            meta = bucket.get("metadata", {})
            if meta.get("type") in ("permanent", "feel", "i", "plan", "letter") or meta.get("pinned") or meta.get("protected"):
                continue
            try:
                scores.append(self.calculate_score(meta))
            except Exception as exc:
                logger.warning("维护周期评分失败: %s: %s", bucket.get("id"), exc)
        result = {
            "checked": len(scores), "archived": 0, "auto_resolved": 0,
            "demoted_orphans": 0,
            "backfilled_embeddings": await self._self_heal_embeddings(buckets),
            "lowest_score": min(scores, default=0),
        }
        logger.info("稳定保存维护周期完成: %s", result)
        return result

    async def _self_heal_embeddings(self, buckets: list) -> int:
        """周期自愈：给「落盘了但 embeddings.db 里没向量」的活跃桶补向量。

        背景（#6）：permanent 桶常因批量导入 / dashboard 钉选而漏建向量，
        breath 的向量通道就检索不到它们，表现为「只读得到 dynamic」。衰减循环
        每轮顺手补齐，无需人工跑 backfill_embeddings.py。

        边界：embedding 未启用 → 跳过；每轮最多补 _BACKFILL_MAX_PER_CYCLE 条
        （防打爆 API），剩余下一轮继续；单条失败仅 warning（rule.md §1.5 允许降级）。
        只处理活跃桶（buckets 不含 archive），不在此删孤儿向量（删除走专用脚本，
        避免把 archive 桶的有效向量误判为孤儿）。"""
        outbox = getattr(self.bucket_mgr, "embedding_outbox", None)
        if outbox is not None and getattr(outbox, "running", False):
            try:
                queued = await outbox.reconcile(
                    buckets=buckets,
                    include_archive=False,
                )
                if queued:
                    logger.info(
                        "Decay self-heal queued / 衰减自愈已加入向量队列: %s 条",
                        queued,
                    )
                return queued
            except Exception as e:
                logger.warning(f"self-heal embeddings: 投递后台队列失败: {e}")
                return 0

        ee = getattr(self.bucket_mgr, "embedding_engine", None)
        if not ee or not getattr(ee, "enabled", False):
            return 0
        try:
            index_ids = set(ee.list_all_ids())
        except Exception as e:
            logger.warning(f"self-heal embeddings: 读取向量索引失败: {e}")
            return 0
        missing = [b for b in buckets if b["id"] not in index_ids and (b.get("content") or "").strip()]
        if not missing:
            return 0
        healed = 0
        for b in missing[:_BACKFILL_MAX_PER_CYCLE]:
            try:
                if await ee.generate_and_store(b["id"], b["content"]):
                    healed += 1
            except Exception as e:
                logger.warning(f"self-heal embeddings: 补 {b['id']} 失败: {e}")
        if healed:
            remaining = len(missing) - healed
            logger.info(
                f"Decay self-heal / 自愈补向量: {healed} 条"
                + (f"（本轮上限 {_BACKFILL_MAX_PER_CYCLE}，剩 {remaining} 下轮继续）"
                   if remaining > 0 else "")
            )
        return healed

    # ---------------------------------------------------------
    # Background decay task management
    # 后台衰减任务管理
    # ---------------------------------------------------------
    async def ensure_started(self) -> None:
        """
        Ensure the decay engine is started (lazy init on first call).
        确保衰减引擎已启动（懒加载，首次调用时启动）。
        """
        if not self._running:
            await self.start()

    async def start(self) -> None:
        """Start the background decay loop.
        启动后台衰减循环。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._background_loop())
        logger.info(
            f"检索索引维护已启动，检查间隔: {self.check_interval} 小时"
        )

    async def stop(self) -> None:
        """Stop the background decay loop.
        停止后台衰减循环。"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("检索索引维护已停止")

    async def _background_loop(self) -> None:
        """后台循环只补齐检索索引，再等待下一个周期。"""
        while self._running:
            try:
                await self.run_decay_cycle()
            except Exception as e:
                logger.error(f"Decay cycle error / 衰减周期出错: {e}")
            # --- Wait for next cycle / 等待下一个周期 ---
            try:
                await asyncio.sleep(self.check_interval * _SECONDS_PER_HOUR)
            except asyncio.CancelledError:
                break
