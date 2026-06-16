"""pimo-ai 异步事件流。

基于 asyncio.Queue 实现的生产者-消费者事件流，用于 Provider 向调用方
推送归一化后的 AssistantMessageEvent。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from pimo.ai.types import AssistantMessage, AssistantMessageEvent

# 私有哨兵：标记流结束，__aiter__ 遇到此对象时退出迭代
_SENTINEL = object()


class AssistantMessageEventStream:
    """基于 asyncio.Queue 的异步事件流。

    Provider（生产者）通过 ``push()`` 推送事件，通过 ``end()`` 标记流结束。
    调用方（消费者）通过 ``async for`` 迭代消费事件，或通过 ``result()``
    等待流结束后获取最终 AssistantMessage。

    使用示例::

        stream = AssistantMessageEventStream()
        # 生产者端（通常在 Provider 内部异步任务中）
        stream.push(event1)
        stream.push(event2)
        stream.end(final_message)

        # 消费者端
        async for event in stream:
            handle(event)
        final = await stream.result()
    """

    def __init__(self) -> None:
        """初始化空事件流。"""
        self._queue: asyncio.Queue[AssistantMessageEvent | object] = asyncio.Queue()
        self._result_future: asyncio.Future[AssistantMessage] = asyncio.Future()
        self._done: bool = False

    def push(self, event: AssistantMessageEvent) -> None:
        """推送一个事件到流中。

        由 Provider 调用。事件按推送顺序被消费者消费。
        流已结束后（``end()`` 已被调用）的推送被静默丢弃。

        若推送的是 DoneEvent 或 ErrorEvent，自动标记流结束并解析
        ``result()``，等效于先 ``push(event)`` 再 ``end(event.message)``。
        其后入队的哨兵确保消费者迭代正常退出。

        Args:
            event: 归一化后的 AssistantMessageEvent。
        """
        if self._done:
            return
        self._queue.put_nowait(event)
        if event.type in ("done", "error"):
            self._done = True
            if not self._result_future.done():
                self._result_future.set_result(event.message)
            self._queue.put_nowait(_SENTINEL)

    def end(self, message: AssistantMessage) -> None:
        """标记流结束并设置最终结果。

        由 Provider 在所有事件推送完毕后调用。调用后消费者无法再接收
        新事件，``result()`` 解析为该 message。重复调用无副作用。

        Args:
            message: 流结束后最终定型的 AssistantMessage。
        """
        if self._done:
            return
        self._done = True
        self._result_future.set_result(message)
        self._queue.put_nowait(_SENTINEL)

    async def __aiter__(self) -> AsyncIterator[AssistantMessageEvent]:
        """异步迭代流中的所有事件。

        消费者通过 ``async for event in stream:`` 使用。
        迭代在流结束且所有已推送事件被消费后自动退出。

        Yields:
            AssistantMessageEvent: 按推送顺序的下一个事件。
        """
        while True:
            event = await self._queue.get()
            if event is _SENTINEL:
                return
            yield event  # type: ignore[attr-defined]

    async def result(self) -> AssistantMessage:
        """等待流结束并获取最终 AssistantMessage。

        可在任何时刻调用（包括迭代前、迭代中、迭代后）。
        若流尚未结束则阻塞等待。

        Returns:
            由 ``end()`` 或终止事件（DoneEvent/ErrorEvent）设置的最终消息。
        """
        return await self._result_future
