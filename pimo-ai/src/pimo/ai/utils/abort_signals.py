"""多取消源合并。

将多个 asyncio.Event 组合为一个复合信号：任一源触发时复合信号也触发。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class CombinedAbortSignal:
    """组合后的取消信号 + 清理函数。"""

    signal: asyncio.Event | None = None
    """复合信号。任一源触发时被 set。无源时为 None。"""

    cleanup: asyncio.Task | None = None
    """后台轮询任务的句柄。调用 cleanup() 可取消任务。"""


def combine_abort_signals(
    signals: list[asyncio.Event | None],
) -> CombinedAbortSignal:
    """将多个 asyncio.Event 组合为一个复合信号。

    - 0 个有效信号 → signal=None
    - 1 个有效信号 → 直接返回该信号
    - 2+ 个信号 → 创建新 Event，后台轮询，任一源触发时 set

    Args:
        signals: 取消信号列表（可能含 None）。

    Returns:
        CombinedAbortSignal: 复合信号 + 清理函数。
    """
    active = [s for s in signals if s is not None]

    if not active:
        return CombinedAbortSignal(signal=None, cleanup=None)

    if len(active) == 1:
        return CombinedAbortSignal(signal=active[0], cleanup=None)

    combined = asyncio.Event()

    async def _watch() -> None:
        """后台轮询所有源信号，任一触发时 set 复合信号。"""
        while not combined.is_set():
            for src in active:
                if src.is_set():
                    combined.set()
                    return
            await asyncio.sleep(0.01)

    task = asyncio.create_task(_watch())

    return CombinedAbortSignal(signal=combined, cleanup=task)
