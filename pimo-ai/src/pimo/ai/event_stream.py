"""pimo-ai 异步事件流。

基于 asyncio.Queue 实现的生产者-消费者事件流，用于 Provider 向调用方
推送归一化后的 AssistantMessageEvent。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from pimo.ai.types import AssistantMessage, AssistantMessageEvent


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
        ...

    def push(self, event: AssistantMessageEvent) -> None:
        """推送一个事件到流中。

        由 Provider 调用。事件按推送顺序被消费者消费。

        Args:
            event: 归一化后的 AssistantMessageEvent。
        """
        ...

    def end(self, message: AssistantMessage) -> None:
        """标记流结束并设置最终结果。

        由 Provider 在所有事件推送完毕后调用。调用后消费者无法再接收
        新事件，``result()`` 解析为该 message。

        Args:
            message: 流结束后最终定型的 AssistantMessage。
        """
        ...

    async def __aiter__(self) -> AsyncIterator[AssistantMessageEvent]:
        """异步迭代流中的所有事件。

        消费者通过 ``async for event in stream:`` 使用。
        迭代在流结束（``end()`` 被调用）且所有已推送事件被消费后自动退出。

        Yields:
            AssistantMessageEvent: 按推送顺序的下一个事件。
        """
        ...

    async def result(self) -> AssistantMessage:
        """等待流结束并获取最终 AssistantMessage。

        可在任何时刻调用（包括迭代前、迭代中、迭代后）。
        若流尚未结束则阻塞等待。

        Returns:
            由 ``end()`` 设置的最终 AssistantMessage。
        """
        ...
