"""跨 Provider 消息格式转换。

在发送消息到特定 Provider 前，对消息列表进行兼容性处理：
- 不支持图片的模型 → 图片降级为文本占位符
- thinking 块跨模型重放 → 转纯文本（同一模型保留）
- tool call ID 规范化（| 分隔符、最大长度等）
- 孤儿 tool call（有调用无结果）→ 合成虚拟 toolResult
- 错误/取消的 assistant 消息 → 跳过（不重放不完整回合）
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pimo.ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
)

# 图片降级占位符
_NON_VISION_USER_IMAGE_PLACEHOLDER = (
    "(image omitted: model does not support images)"
)
_NON_VISION_TOOL_IMAGE_PLACEHOLDER = (
    "(tool image omitted: model does not support images)"
)


# =============================================================================
# 公共函数
# =============================================================================


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id: Callable[
        [str, Model, Message], str
    ] | None = None,
) -> list[Message]:
    """对消息列表进行跨 Provider 兼容性转换。

    两遍扫描:
    1. 图片降级 + thinking 块处理 + tool call ID 规范化
    2. 孤儿 tool call 合成虚拟 toolResult + 跳过错误回合

    Args:
        messages: 原始消息列表。
        model: 目标模型（用于判断是否支持图片、是否是同一模型）。
        normalize_tool_call_id: tool call ID 规范化回调。

    Returns:
        转换后的消息列表。
    """
    # 原始 ID → 规范化 ID 映射
    tool_call_id_map: dict[str, str] = {}

    # ---- Pass 0: 图片降级 ----
    image_aware = _downgrade_unsupported_images(messages, model)

    # ---- Pass 1: 逐消息转换 ----
    transformed: list[Message] = []
    for msg in image_aware:
        if msg.role == "user":
            transformed.append(msg)

        elif msg.role == "toolResult":
            normalized_id = tool_call_id_map.get(msg.tool_call_id)
            if normalized_id and normalized_id != msg.tool_call_id:
                new_msg = ToolResultMessage(
                    role="toolResult",
                    tool_call_id=normalized_id,
                    tool_name=msg.tool_name,
                    content=msg.content,
                    is_error=msg.is_error,
                    timestamp=msg.timestamp,
                )
                transformed.append(new_msg)
            else:
                transformed.append(msg)

        elif msg.role == "assistant":
            assistant_msg = msg
            is_same_model = (
                assistant_msg.provider == model.provider
                and assistant_msg.api == model.api
                and assistant_msg.model == model.id
            )

            new_content: list = []
            for block in assistant_msg.content:
                if isinstance(block, ThinkingContent):
                    # redacted thinking 跨模型 → 删除
                    if block.redacted:
                        if is_same_model:
                            new_content.append(block)
                        continue
                    # 同一模型 + 有签名 → 保留（用于重放）
                    if is_same_model and block.signature:
                        new_content.append(block)
                        continue
                    # 空 thinking → 跳过
                    if not block.thinking or not block.thinking.strip():
                        continue
                    # 同一模型 → 保留，跨模型 → 转 text
                    if is_same_model:
                        new_content.append(block)
                    else:
                        new_content.append(
                            TextContent(text=block.thinking)
                        )

                elif isinstance(block, TextContent):
                    new_content.append(block)

                elif isinstance(block, ToolCall):
                    tc = block
                    # 跨模型：清理 thoughtSignature
                    if (
                        not is_same_model
                        and hasattr(tc, "thoughtSignature")
                    ):
                        delattr(tc, "thoughtSignature")

                    # 跨模型 + 有规范化器：规范化 ID
                    if not is_same_model and normalize_tool_call_id:
                        new_id = normalize_tool_call_id(
                            tc.id, model, assistant_msg,
                        )
                        if new_id != tc.id:
                            tool_call_id_map[tc.id] = new_id
                            tc.id = new_id

                    new_content.append(tc)

                else:
                    new_content.append(block)

            transformed.append(
                AssistantMessage(
                    role="assistant",
                    content=new_content,
                    api=assistant_msg.api,
                    provider=assistant_msg.provider,
                    model=assistant_msg.model,
                    usage=assistant_msg.usage,
                    stop_reason=assistant_msg.stop_reason,
                    error_message=assistant_msg.error_message,
                    response_id=assistant_msg.response_id,
                    timestamp=assistant_msg.timestamp,
                )
            )

        else:
            transformed.append(msg)

    # ---- Pass 2: 孤儿 tool call 合成 ----
    result: list[Message] = []
    pending_calls: list[ToolCall] = []
    existing_result_ids: set[str] = set()

    def _flush_orphans() -> None:
        if not pending_calls:
            return
        for tc in pending_calls:
            if tc.id not in existing_result_ids:
                result.append(
                    ToolResultMessage(
                        role="toolResult",
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=[
                            TextContent(text="No result provided")
                        ],
                        is_error=True,
                        timestamp=time.time(),
                    )
                )
        pending_calls.clear()
        existing_result_ids.clear()

    for msg in transformed:
        if msg.role == "assistant":
            _flush_orphans()

            # 跳过 error/aborted 的不完整回合
            if msg.stop_reason in ("error", "aborted"):
                continue

            # 记录本轮 tool calls
            tool_calls = [
                b for b in msg.content if isinstance(b, ToolCall)
            ]
            if tool_calls:
                pending_calls = list(tool_calls)
                existing_result_ids = set()

            result.append(msg)

        elif msg.role == "toolResult":
            existing_result_ids.add(msg.tool_call_id)
            result.append(msg)

        elif msg.role == "user":
            # user 消息中断工具调用流程 → 合成剩余孤儿
            _flush_orphans()
            result.append(msg)

        else:
            result.append(msg)

    # 末尾可能残留孤儿
    _flush_orphans()

    return result


# =============================================================================
# 内部辅助
# =============================================================================


def _replace_images_with_placeholder(
    content: list[TextContent | ImageContent],
    placeholder: str,
) -> list[TextContent]:
    """将内容列表中的图片替换为文本占位符。

    连续多张图片只生成一个占位符，避免重复。

    Args:
        content: 原始内容块列表。
        placeholder: 占位符文本。

    Returns:
        仅含 TextContent 的列表。
    """
    result: list[TextContent] = []
    previous_was_placeholder = False

    for block in content:
        if isinstance(block, ImageContent):
            if not previous_was_placeholder:
                result.append(TextContent(text=placeholder))
            previous_was_placeholder = True
            continue

        result.append(block)
        # 检测已有的占位符文本（避免与已有文本中恰好相同的文字冲突时重复去重）
        if isinstance(block, TextContent):
            previous_was_placeholder = block.text == placeholder
        else:
            previous_was_placeholder = False

    return result


def _downgrade_unsupported_images(
    messages: list[Message],
    model: Model,
) -> list[Message]:
    """对不支持图片输入的模型降级图片为占位符。

    仅处理 user 和 toolResult 角色中的图片。

    Args:
        messages: 原始消息列表。
        model: 目标模型。

    Returns:
        图片已降级的消息列表。
    """
    if "image" in model.input_types:
        return list(messages)

    result: list[Message] = []
    for msg in messages:
        if msg.role == "user" and isinstance(msg.content, list):
            new_content = _replace_images_with_placeholder(
                msg.content, _NON_VISION_USER_IMAGE_PLACEHOLDER,
            )
            result.append(
                type(msg)(
                    **{
                        **msg.__dict__,
                        "content": new_content,
                    }
                )
            )
        elif msg.role == "toolResult":
            new_content = _replace_images_with_placeholder(
                msg.content, _NON_VISION_TOOL_IMAGE_PLACEHOLDER,
            )
            result.append(
                ToolResultMessage(
                    role="toolResult",
                    tool_call_id=msg.tool_call_id,
                    tool_name=msg.tool_name,
                    content=new_content,
                    is_error=msg.is_error,
                    timestamp=msg.timestamp,
                )
            )
        else:
            result.append(msg)

    return result
