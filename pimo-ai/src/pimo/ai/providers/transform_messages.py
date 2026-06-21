"""跨 Provider 消息格式转换（占位）。

TODO: 完整实现将在后续 /gen 任务中完成。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pimo.ai.types import Message, Model


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id: Callable[[str, Model, Message], str] | None = None,
) -> list[Message]:
    """在发送前规范化跨 Provider 消息格式。

    处理:
    - tool call ID 规范化（| 分隔符、最大长度限制）
    - 跨 provider 消息兼容性调整

    Args:
        messages: 统一格式的消息列表。
        model: 目标模型。
        normalize_tool_call_id: 可选的 tool call ID 规范化回调。

    Returns:
        规范化后的消息列表。
    """
    ...
