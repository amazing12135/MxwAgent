"""Anthropic Messages 协议 Provider。

通过 Anthropic Python SDK 调用 Anthropic Messages API，将厂商原生 SSE 流
归一化为 pimo-ai 统一事件流。

支持: 标准 API Key、自定义 base_url、自适应思考 (adaptive thinking)、
预算式思考 (budget-based thinking)、提示缓存 (ephemeral cache_control)。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

import anthropic

from pimo.ai.api_registry import ApiProvider
from pimo.ai.event_stream import AssistantMessageEventStream
from pimo.ai.models.cost import calculate_cost
from pimo.ai.types import (
    AssistantMessage,
    Context,
    CostInfo,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolResultMessage,
    Usage,
)

# =============================================================================
# Anthropic 特有类型
# =============================================================================

AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
"""自适应思考的努力级别。

- ``"max"``: 始终思考，无约束（仅 Opus 4.6）
- ``"xhigh"``: 最高推理级别（Opus 4.7+、Fable 5）
- ``"high"``: 始终思考，深层推理
- ``"medium"``: 中等思考，简单查询可能跳过
- ``"low"``: 最小思考，大多数简单任务跳过
"""

AnthropicThinkingDisplay = Literal["summarized", "omitted"]
"""思考内容在 API 响应中的返回方式。

- ``"summarized"``: 思考块包含总结文本（默认，与旧 Claude 4 行为一致）
- ``"omitted"``: 思考块返回空字段，签名仍回传以维持多轮连续性。
  适用于不需要展示思考内容的 UI 场景。
"""

# Beta feature identifiers
_FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
_INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"


@dataclass(kw_only=True)
class AnthropicOptions(StreamOptions):
    """Anthropic Messages API 扩展选项。

    在 StreamOptions 基础上增加 Anthropic 特有的 thinking、tool_choice
    和预构建 client 参数。
    """

    thinking_enabled: bool | None = None
    """启用 extended thinking。

    - ``True``: 开启思考（自适应模型用 effort 控制，旧模型用 budget）
    - ``False``: 显式禁用思考
    - ``None``: 省略 thinking 参数，使用 API 默认行为
    """

    thinking_budget_tokens: int | None = None
    """token 预算（仅旧模型，自适应模型忽略）。
    默认: 1024 (当 thinking_enabled=True 且未提供 budget 时)
    """

    effort: AnthropicEffort | None = None
    """自适应思考模型的努力级别。控制 Claude 分配多少思考量。
    旧模型忽略。
    """

    thinking_display: AnthropicThinkingDisplay | None = None
    """思考内容返回方式。默认: ``"summarized"`` (当 thinking 开启时)。
    """

    interleaved_thinking: bool | None = None
    """是否请求交错思考 beta header。
    自适应思考模型内置交错思考，对此类模型忽略此设置。
    默认: True。
    """

    tool_choice: str | dict[str, str] | None = None
    """工具选择行为。

    - ``"auto"`` | ``"any"`` | ``"none"``: Anthropic 内置选择
    - ``{"type": "tool", "name": "..."}``: 强制指定工具
    - ``None``: 省略（Anthropic 默认行为，当前等价于 auto）
    """

    client: Any = None
    """预构建的 Anthropic SDK 客户端实例。

    非 None 时跳过内部客户端构造。可用于注入 ``AnthropicVertex`` 等
    共享 Messages API 的替代客户端。
    """


# =============================================================================
# AnthropicProvider
# =============================================================================


class AnthropicProvider(ApiProvider):
    """Anthropic Messages API Provider。

    封装 Anthropic Python SDK，支持标准 API Key 和自定义 base_url。
    通过 SDK 内置 streaming 迭代事件，逐条转换为归一化事件推入 stream。

    自适应思考模型的 effort 映射逻辑:
    - thinkingLevelMap 中有显式映射 → 使用映射值
    - 否则根据 ThinkingLevel 推算: minimal/low→"low", medium→"medium", high→"high"
    """

    api: str = "anthropic-messages"

    # -------------------------------------------------------------------------
    # ApiProvider 接口实现
    # -------------------------------------------------------------------------

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """流式调用 Anthropic Messages API。

        Args:
            model: 目标模型描述（api="anthropic-messages"）。
            context: 统一上下文（system_prompt + messages + tools）。
            options: 流式选项。

        Returns:
            AssistantMessageEventStream: 归一化后的异步事件流。
        """
        stream = AssistantMessageEventStream()
        asyncio.create_task(
            self._run_stream(model, context, options, stream)
        )
        return stream

    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """stream 的简化版本，额外支持 reasoning 参数。

        - reasoning=None: 禁用 thinking，直接委托给 stream()
        - reasoning != None 且自适应模型: 将 reasoning 映射为 effort
        - reasoning != None 且旧模型: 用 budget-based thinking

        Args:
            model: 目标模型描述。
            context: 统一上下文。
            options: 包含 reasoning、session_id、thinking_budgets 的选项。

        Returns:
            AssistantMessageEventStream: 归一化后的异步事件流。
        """
        api_key = options.api_key if options else None
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        # 构建基参：将 SimpleStreamOptions 展开为 StreamOptions
        base = StreamOptions(
            temperature=options.temperature if options else None,
            max_tokens=options.max_tokens if options else None,
            signal=options.signal if options else None,
            api_key=api_key,
            transport=options.transport if options else "auto",
            cache_retention=options.cache_retention if options else None,
            session_id=options.session_id if options else None,
            headers=options.headers if options else None,
            timeout_ms=options.timeout_ms if options else None,
            max_retries=options.max_retries if options else None,
            max_retry_delay_ms=options.max_retry_delay_ms if options else None,
            on_payload=options.on_payload if options else None,
            on_response=options.on_response if options else None,
            metadata=options.metadata if options else None,
        )

        reasoning = options.reasoning if options else None
        if not reasoning:
            # 无 reasoning → 禁用 thinking
            return await self.stream(
                model,
                context,
                AnthropicOptions(**base.__dict__, thinking_enabled=False),
            )

        # 自适应模型: 将 reasoning 映射为 effort
        if model.compat and model.compat.get("forceAdaptiveThinking"):
            effort = _map_thinking_level_to_effort(model, reasoning)
            return await self.stream(
                model,
                context,
                AnthropicOptions(
                    **base.__dict__,
                    thinking_enabled=True,
                    effort=effort,
                ),
            )

        # 旧模型: budget-based thinking
        adjusted = _adjust_max_tokens_for_thinking(
            base_max_tokens=base.max_tokens,
            model_max_tokens=model.max_tokens,
            reasoning_level=reasoning,
            custom_budgets=options.thinking_budgets if options else None,
        )
        return await self.stream(
            model,
            context,
            AnthropicOptions(
                **base.__dict__,
                max_tokens=adjusted["max_tokens"],
                thinking_enabled=True,
                thinking_budget_tokens=adjusted["thinking_budget"],
            ),
        )

    # -------------------------------------------------------------------------
    # 核心流式执行
    # -------------------------------------------------------------------------

    async def _run_stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None,
        stream: AssistantMessageEventStream,
    ) -> None:
        """在后台异步执行流式调用，将事件推入 stream。

        Args:
            model: 目标模型。
            context: 统一上下文。
            options: 流式选项。
            stream: 事件流。
        """
        output = AssistantMessage(
            role="assistant",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(
                input=0,
                output=0,
                cache_read=0,
                cache_write=0,
                total_tokens=0,
                cost=CostInfo(),
            ),
            stop_reason="stop",
            error_message=None,
            timestamp=time.time(),
        )

        try:
            opts = (
                AnthropicOptions(**options.__dict__)
                if options and not isinstance(options, AnthropicOptions)
                else options
            ) if options else None

            # 1. 构造客户端
            if opts and opts.client:
                client = opts.client
            else:
                api_key = opts.api_key if opts else None
                if not api_key:
                    raise ValueError(
                        f"No API key for provider: {model.provider}"
                    )
                client = _create_client(
                    model=model,
                    api_key=api_key,
                    interleaved_thinking=(
                        opts.interleaved_thinking if opts else True
                    ),
                    use_fine_grained_tool_streaming=_should_use_fine_grained_beta(
                        model, context
                    ),
                    options_headers=opts.headers if opts else None,
                    session_id=opts.session_id if opts else None,
                )

            # 2. 构建请求参数
            params = _build_params(model, context, opts)

            # 3. onPayload 回调
            if opts and opts.on_payload:
                opts.on_payload(params)

            # 4. 发起流式请求
            async with client.messages.stream(**params) as sdk_stream:
                # 推送 start 事件
                stream.push(StartEvent(message=output))

                # 5. 迭代事件
                async for event in sdk_stream:
                    if opts and opts.signal:
                        if getattr(opts.signal, "is_set", lambda: False)():
                            raise asyncio.CancelledError("Request was aborted")

                    if event.type == "message_start":
                        output.response_id = event.message.id
                        sdk_usage = event.message.usage
                        output.usage.input = sdk_usage.input_tokens or 0
                        output.usage.output = sdk_usage.output_tokens or 0
                        output.usage.cache_read = (
                            sdk_usage.cache_read_input_tokens or 0
                        )
                        output.usage.cache_write = (
                            sdk_usage.cache_creation_input_tokens or 0
                        )
                        output.usage.total_tokens = (
                            output.usage.input
                            + output.usage.output
                            + output.usage.cache_read
                            + output.usage.cache_write
                        )
                        calculate_cost(model, output.usage)

                    elif event.type == "content_block_start":
                        _handle_content_block_start(event, output, stream)

                    elif event.type == "content_block_delta":
                        _handle_content_block_delta(event, output, stream)

                    elif event.type == "content_block_stop":
                        _handle_content_block_stop(event, output, stream)

                    elif event.type == "message_delta":
                        if event.delta.stop_reason:
                            sr, err = _map_stop_reason(
                                event.delta.stop_reason
                            )
                            output.stop_reason = sr
                            if err:
                                output.error_message = err
                        if event.usage:
                            if event.usage.output_tokens is not None:
                                output.usage.output = event.usage.output_tokens
                            # Recompute total
                            output.usage.total_tokens = (
                                output.usage.input
                                + output.usage.output
                                + output.usage.cache_read
                                + output.usage.cache_write
                            )
                        # 仅在 message_delta 阶段触发成本计算
                        calculate_cost(model, output.usage)

            # 6. 检查取消
            if opts and opts.signal:
                if getattr(opts.signal, "is_set", lambda: False)():
                    raise asyncio.CancelledError("Request was aborted")

            # 7. 检查终止原因
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(
                    output.error_message or "An unknown error occurred"
                )

            # 8. 推送完成事件
            stream.push(DoneEvent(message=output))

        except (asyncio.CancelledError, Exception) as exc:
            # 清理块上的临时字段
            for block in output.content:
                if hasattr(block, "_index"):
                    delattr(block, "_index")
                if hasattr(block, "_partial_json"):
                    delattr(block, "_partial_json")

            is_aborted = (
                opts
                and opts.signal
                and getattr(opts.signal, "is_set", lambda: False)()
            )
            output.stop_reason = "aborted" if is_aborted else "error"
            output.error_message = str(exc)

            stream.push(ErrorEvent(message=output))


# =============================================================================
# 事件处理 —— 将 Anthropic 原生事件转为 pimo 统一事件
# =============================================================================


def _handle_content_block_start(
    event: Any, output: AssistantMessage, stream: AssistantMessageEventStream
) -> None:
    """处理 content_block_start 事件，向 output 追加新内容块。"""
    cb = event.content_block
    idx = event.index

    if cb.type == "text":
        block = TextContent(text="")
        setattr(block, "_index", idx)
        output.content.append(block)
        stream.push(StartEvent(message=output))

    elif cb.type == "thinking":
        block = ThinkingContent(thinking="", signature="")
        setattr(block, "_index", idx)
        output.content.append(block)
        stream.push(StartEvent(message=output))

    elif cb.type == "redacted_thinking":
        block = ThinkingContent(
            thinking="[Reasoning redacted]",
            signature=cb.data,
            redacted=True,
        )
        setattr(block, "_index", idx)
        output.content.append(block)
        stream.push(StartEvent(message=output))

    elif cb.type == "tool_use":
        block = ToolCall(
            id=cb.id,
            name=cb.name,
            arguments=cb.input if cb.input else {},
        )
        setattr(block, "_index", idx)
        setattr(block, "_partial_json", "")
        output.content.append(block)
        stream.push(StartEvent(message=output))


def _handle_content_block_delta(
    event: Any, output: AssistantMessage, stream: AssistantMessageEventStream
) -> None:
    """处理 content_block_delta 事件，更新 content 列表中对应块。"""
    delta = event.delta
    idx = event.index
    block = _find_block_by_index(output.content, idx)

    if not block:
        return

    if delta.type == "text_delta":
        if isinstance(block, TextContent):
            block.text += delta.text
            stream.push(TextDeltaEvent(message=output))

    elif delta.type == "thinking_delta":
        if isinstance(block, ThinkingContent):
            block.thinking += delta.thinking
            stream.push(ThinkingDeltaEvent(message=output))

    elif delta.type == "signature_delta":
        if isinstance(block, ThinkingContent):
            block.signature = (block.signature or "") + delta.signature

    elif delta.type == "input_json_delta":
        if isinstance(block, ToolCall):
            partial = getattr(block, "_partial_json", "")
            partial += delta.partial_json
            setattr(block, "_partial_json", partial)
            # 流式解析 JSON 参数（简化：最终才做完整解析）
            stream.push(ToolCallDeltaEvent(message=output))


def _handle_content_block_stop(
    event: Any, output: AssistantMessage, stream: AssistantMessageEventStream
) -> None:
    """处理 content_block_stop 事件，清理临时字段并推送最终状态。"""
    idx = event.index
    block = _find_block_by_index(output.content, idx)

    if not block:
        return

    # 清理临时 _index 标记
    if hasattr(block, "_index"):
        delattr(block, "_index")

    if isinstance(block, ToolCall):
        # 最终解析 JSON 参数
        partial = getattr(block, "_partial_json", "")
        if hasattr(block, "_partial_json"):
            delattr(block, "_partial_json")
        try:
            block.arguments = json.loads(partial) if partial.strip() else {}
        except json.JSONDecodeError:
            block.arguments = {}

    # 推送块完成后的最终状态
    stream.push(StartEvent(message=output))


def _find_block_by_index(
    content: list, idx: int
) -> TextContent | ThinkingContent | ToolCall | None:
    """按 Anthropic 事件 index 在 content 列表中查找对应块。"""
    for block in content:
        if getattr(block, "_index", None) == idx:
            return block
    return None


# =============================================================================
# 内部辅助函数
# =============================================================================


def _resolve_cache_retention(cache_retention: str | None = None) -> str:
    """解析缓存保留策略。

    优先级: 显式参数 > PI_CACHE_RETENTION 环境变量 > 默认 "short"

    Args:
        cache_retention: 显式指定的保留策略。

    Returns:
        "none" | "short" | "long"
    """
    if cache_retention:
        return cache_retention
    if os.environ.get("PI_CACHE_RETENTION") == "long":
        return "long"
    return "short"


def _get_cache_control(
    model: Model, cache_retention: str | None = None
) -> dict[str, Any]:
    """计算缓存控制配置。

    Args:
        model: 目标模型。
        cache_retention: 缓存保留策略，为 None 时调用 _resolve_cache_retention 解析。

    Returns:
        {"retention": str, "cache_control": dict | None}
    """
    retention = _resolve_cache_retention(cache_retention)
    if retention == "none":
        return {"retention": retention, "cache_control": None}

    compat = _get_anthropic_compat(model)
    ttl = (
        "1h"
        if retention == "long" and compat.get("supports_long_cache_retention", True)
        else None
    )
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        cache_control["ttl"] = ttl
    return {"retention": retention, "cache_control": cache_control}


def _get_anthropic_compat(model: Model) -> dict[str, bool]:
    """读取 model.compat 中的 Anthropic 兼容性标志并提供默认值。

    自动检测 provider 类型并覆写默认值（如 Fireworks 不支持部分功能）。

    Args:
        model: 目标模型。

    Returns:
        各兼容性标志的完整 dict。
    """
    is_fireworks = model.provider == "fireworks"

    compat = model.compat or {}
    return {
        "supports_eager_tool_input_streaming": compat.get(
            "supportsEagerToolInputStreaming", not is_fireworks
        ),
        "supports_long_cache_retention": compat.get(
            "supportsLongCacheRetention", not is_fireworks
        ),
        "send_session_affinity_headers": compat.get(
            "sendSessionAffinityHeaders", is_fireworks
        ),
        "supports_cache_control_on_tools": compat.get(
            "supportsCacheControlOnTools", not is_fireworks
        ),
        "supports_temperature": compat.get("supportsTemperature", True),
        "allow_empty_signature": compat.get("allowEmptySignature", False),
    }


def _convert_content_blocks(
    content: list[TextContent | ImageContent],
) -> str | list[dict[str, Any]]:
    """将统一的 content blocks 转换为 Anthropic API 格式。

    - 纯文本 → 返回拼接后的字符串
    - 含图片 → 返回 ContentBlockParam 数组（仅图片时自动添加占位文本）

    Args:
        content: TextContent / ImageContent 列表。

    Returns:
        str | list[dict]: Anthropic content 格式。
    """
    has_images = any(c.type == "image" for c in content)

    if not has_images:
        return "".join(
            c.text for c in content if c.type == "text"
        )

    blocks: list[dict[str, Any]] = []
    for c in content:
        if c.type == "text":
            blocks.append({"type": "text", "text": c.text})
        elif c.type == "image":
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": c.mime_type,
                    "data": c.data,
                },
            })

    has_text = any(b["type"] == "text" for b in blocks)
    if not has_text:
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})

    return blocks


def _normalize_tool_call_id(tool_call_id: str) -> str:
    """规范化 tool_use id 以匹配 Anthropic 格式约束。

    - 替换非法字符为 ``_``
    - 截断到 64 字符

    Args:
        tool_call_id: 原始 tool call id。

    Returns:
        规范化后的 id。
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)[:64]


def _convert_messages(
    messages: list[Message],
    model: Model,
    cache_control: dict[str, Any] | None = None,
    *,
    allow_empty_signature: bool = False,
) -> list[dict[str, Any]]:
    """将统一 Message 列表转换为 Anthropic Messages API 格式。

    Args:
        messages: 统一格式的消息列表。
        model: 目标模型。
        cache_control: 缓存控制对象，注入到最后一条 user message。
        allow_empty_signature: True 时保留空签名 thinking 块。

    Returns:
        Anthropic MessageCreateParams 所需的消息列表。
    """
    params: list[dict[str, Any]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if msg.role == "user":
            if isinstance(msg.content, str):
                params.append({
                    "role": "user",
                    "content": msg.content,
                })
            elif isinstance(msg.content, list):
                # 纯文本块列表 → 合并为单个字符串
                text_parts = [
                    c.text for c in msg.content if c.type == "text" and c.text.strip()
                ]
                has_images = any(c.type == "image" for c in msg.content)

                if not has_images and text_parts:
                    params.append({
                        "role": "user",
                        "content": "\n".join(text_parts),
                    })
                elif has_images or text_parts:
                    blocks: list[dict[str, Any]] = []
                    for c in msg.content:
                        if c.type == "text":
                            if c.text.strip():
                                blocks.append({"type": "text", "text": c.text})
                        elif c.type == "image":
                            blocks.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": c.mime_type,
                                    "data": c.data,
                                },
                            })
                    if blocks:
                        params.append({"role": "user", "content": blocks})

        elif msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    if block.text.strip():
                        blocks.append({
                            "type": "text",
                            "text": block.text,
                        })
                elif isinstance(block, ThinkingContent):
                    if block.redacted:
                        blocks.append({
                            "type": "redacted_thinking",
                            "data": block.signature or "",
                        })
                    elif block.thinking.strip():
                        if not block.signature or not block.signature.strip():
                            # 无签名 → 转为 text（除非允许空签名）
                            if allow_empty_signature:
                                blocks.append({
                                    "type": "thinking",
                                    "thinking": block.thinking,
                                    "signature": "",
                                })
                            else:
                                blocks.append({
                                    "type": "text",
                                    "text": block.thinking,
                                })
                        else:
                            blocks.append({
                                "type": "thinking",
                                "thinking": block.thinking,
                                "signature": block.signature,
                            })
                elif isinstance(block, ToolCall):
                    blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.arguments,
                    })
            if blocks:
                params.append({"role": "assistant", "content": blocks})

        elif msg.role == "toolResult":
            tool_results: list[dict[str, Any]] = []

            # 收集连续的 toolResult
            j = i
            while j < len(messages) and messages[j].role == "toolResult":
                tr = messages[j]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": _convert_content_blocks(tr.content),
                    "is_error": tr.is_error,
                })
                j += 1

            # 合并到单个 user message
            params.append({
                "role": "user",
                "content": tool_results,
            })
            i = j - 1  # 跳过已处理的连续 toolResult

        i += 1

    # 在最后一条 user message 上注入 cache_control
    if cache_control and params:
        last = params[-1]
        if last["role"] == "user":
            if isinstance(last["content"], list):
                last_block = last["content"][-1]
                if last_block and last_block.get("type") in (
                    "text", "image", "tool_result",
                ):
                    last_block["cache_control"] = cache_control
            elif isinstance(last["content"], str):
                last["content"] = [{
                    "type": "text",
                    "text": last["content"],
                    "cache_control": cache_control,
                }]

    return params


def _convert_tools(
    tools: list[dict[str, Any]],
    supports_eager_input_streaming: bool = True,
    cache_control: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """将统一 Tool JSON Schema 列表转换为 Anthropic Tool 格式。

    Args:
        tools: JSON Schema 格式的工具定义列表。
        supports_eager_input_streaming: True 时添加 eager_input_streaming 字段。
        cache_control: 缓存控制对象，注入到最后一个 tool。

    Returns:
        Anthropic Tool 数组。
    """
    if not tools:
        return []

    result: list[dict[str, Any]] = []
    for i, tool in enumerate(tools):
        schema = tool.get("parameters", {}) if isinstance(tool, dict) else {}
        converted: dict[str, Any] = {
            "name": tool.get("name", "") if isinstance(tool, dict) else "",
            "description": tool.get("description", "") if isinstance(tool, dict) else "",
            "input_schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        }

        if supports_eager_input_streaming:
            converted["eager_input_streaming"] = True

        if cache_control and i == len(tools) - 1:
            converted["cache_control"] = cache_control

        result.append(converted)

    return result


def _map_stop_reason(
    reason: str, stop_details: dict[str, Any] | None = None
) -> tuple[str, str | None]:
    """将 Anthropic stop_reason 映射为 pimo-ai 标准 stop_reason。

    映射:
        end_turn / pause_turn / stop_sequence → "stop"
        max_tokens → "length"
        tool_use → "toolUse"
        refusal → "error" + explanation
        sensitive → "error"

    Args:
        reason: Anthropic stop_reason 字符串。
        stop_details: stop_reason 附加详情。

    Returns:
        (standardized_stop_reason, optional_error_message)
    """
    if reason == "end_turn":
        return "stop", None
    elif reason == "max_tokens":
        return "length", None
    elif reason == "tool_use":
        return "toolUse", None
    elif reason == "refusal":
        explanation = (
            stop_details.get("explanation")
            if stop_details
            else "The model refused to complete the request"
        ) or "The model refused to complete the request"
        return "error", explanation
    elif reason == "pause_turn":
        return "stop", None
    elif reason == "stop_sequence":
        return "stop", None
    elif reason == "sensitive":
        return "error", "Content flagged by safety filters"
    else:
        raise ValueError(f"Unhandled stop reason: {reason}")


def _map_thinking_level_to_effort(
    model: Model, level: str | None
) -> AnthropicEffort:
    """将 pimo-ai ThinkingLevel 映射为 Anthropic adaptive thinking effort。

    优先使用 model.thinking_level_map 中的显式映射。

    Args:
        model: 目标模型。
        level: pimo-ai ThinkingLevel。

    Returns:
        对应的 AnthropicEffort。
    """
    if level and model.thinking_level_map:
        mapped = model.thinking_level_map.get(level)
        if isinstance(mapped, str):
            return mapped  # type: ignore[return-value]

    # 默认映射
    if level in ("minimal", "low"):
        return "low"
    elif level == "medium":
        return "medium"
    else:
        return "high"


def _adjust_max_tokens_for_thinking(
    base_max_tokens: int | None,
    model_max_tokens: int,
    reasoning_level: str,
    custom_budgets: dict[str, int] | None = None,
) -> dict[str, int]:
    """为非自适应模型计算 max_tokens 和 thinking token 预算。

    将 thinking budget 嵌入 max_tokens 空间，当两者接近时保留最小输出容量。

    Args:
        base_max_tokens: 调用方显式设置的 max_tokens（None 则用模型上限）。
        model_max_tokens: 模型最大输出 token 数。
        reasoning_level: pimo-ai ThinkingLevel。
        custom_budgets: per-level thinking token 预算覆写。

    Returns:
        {"max_tokens": int, "thinking_budget": int}
    """
    default_budgets = {
        "minimal": 1024,
        "low": 2048,
        "medium": 8192,
        "high": 16384,
    }
    budgets = {**default_budgets, **(custom_budgets or {})}

    # clamp xhigh → high
    level = "high" if reasoning_level == "xhigh" else reasoning_level
    thinking_budget = budgets.get(level, 1024)

    min_output_tokens = 1024
    max_tokens = (
        model_max_tokens
        if base_max_tokens is None
        else min(base_max_tokens + thinking_budget, model_max_tokens)
    )

    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - min_output_tokens)

    return {"max_tokens": max_tokens, "thinking_budget": thinking_budget}


# =============================================================================
# SDK 客户端构造
# =============================================================================


def _create_client(
    model: Model,
    api_key: str,
    *,
    interleaved_thinking: bool = True,
    use_fine_grained_tool_streaming: bool = False,
    options_headers: dict[str, str] | None = None,
    session_id: str | None = None,
) -> anthropic.Anthropic:
    """构造 Anthropic SDK 客户端。

    根据 model.compat 自动配置 beta header 和 session affinity。

    Args:
        model: 目标模型。
        api_key: API Key。
        interleaved_thinking: 是否启用 interleaved thinking beta。
        use_fine_grained_tool_streaming: 是否启用 fine-grained tool streaming beta。
        options_headers: 调用方附加 HTTP 头。
        session_id: 缓存亲和会话 ID。

    Returns:
        配置完成的 Anthropic SDK 客户端实例。
    """
    compat = _get_anthropic_compat(model)

    # Beta features
    beta_features: list[str] = []
    if use_fine_grained_tool_streaming:
        beta_features.append(_FINE_GRAINED_TOOL_STREAMING_BETA)
    if interleaved_thinking:
        # 自适应模型不需要 interleaved thinking beta
        is_adaptive = model.compat and model.compat.get("forceAdaptiveThinking")
        if not is_adaptive:
            beta_features.append(_INTERLEAVED_THINKING_BETA)

    # 构建 default_headers
    default_headers: dict[str, str] = {
        "anthropic-dangerous-direct-browser-access": "true",
    }

    if beta_features:
        default_headers["anthropic-beta"] = ",".join(beta_features)

    # Session affinity header
    if session_id and compat.get("send_session_affinity_headers"):
        default_headers["x-session-affinity"] = session_id

    # 合并 model.headers 和 options_headers
    if model.compat and isinstance(model.compat, dict):
        model_headers = {
            k: v for k, v in model.compat.items()
            if k.startswith("header:")
        }
        for k, v in model_headers.items():
            header_name = k[len("header:"):]
            default_headers[header_name] = str(v)

    if options_headers:
        default_headers.update(options_headers)

    return anthropic.Anthropic(
        api_key=api_key,
        base_url=model.base_url,
        default_headers=default_headers,
        max_retries=0,  # pi 默认不重试，由上层控制
    )


def _should_use_fine_grained_beta(
    model: Model, context: Context
) -> bool:
    """判断是否需要启用 fine-grained-tool-streaming beta。

    仅当模型不支持 eager_input_streaming 且有工具时启用。

    Args:
        model: 目标模型。
        context: 统一上下文。

    Returns:
        True 时应启用 beta header。
    """
    compat = _get_anthropic_compat(model)
    has_tools = bool(context.tools and len(context.tools) > 0)
    return has_tools and not compat.get("supports_eager_tool_input_streaming", True)


# =============================================================================
# 请求参数构建
# =============================================================================


def _build_params(
    model: Model,
    context: Context,
    options: AnthropicOptions | None = None,
) -> dict[str, Any]:
    """构建 Anthropic Messages API 请求参数。

    Args:
        model: 目标模型。
        context: 统一上下文。
        options: Anthropic 扩展选项。

    Returns:
        Anthropic MessageCreateParams 兼容的参数字典。
    """
    cache = _get_cache_control(model, options.cache_retention if options else None)
    cache_control = cache.get("cache_control")
    compat = _get_anthropic_compat(model)

    params: dict[str, Any] = {
        "model": model.id,
        "max_tokens": (
            options.max_tokens
            if (options and options.max_tokens is not None)
            else model.max_tokens
        ),
    }

    # Messages
    params["messages"] = _convert_messages(
        context.messages, model, cache_control,
        allow_empty_signature=compat.get("allow_empty_signature", False),
    )

    # System prompt
    if context.system_prompt:
        params["system"] = [
            {
                "type": "text",
                "text": context.system_prompt,
                **({"cache_control": cache_control} if cache_control else {}),
            }
        ]

    # Temperature（与 thinking 互斥）
    if (
        options
        and options.temperature is not None
        and not options.thinking_enabled
        and compat.get("supports_temperature", True)
    ):
        params["temperature"] = options.temperature

    # Tools
    if context.tools and len(context.tools) > 0:
        params["tools"] = _convert_tools(
            context.tools,
            supports_eager_input_streaming=compat.get(
                "supports_eager_tool_input_streaming", True
            ),
            cache_control=(
                cache_control
                if compat.get("supports_cache_control_on_tools", True)
                else None
            ),
        )

    # Thinking 配置
    if model.reasoning and options:
        if options.thinking_enabled:
            display = options.thinking_display or "summarized"
            is_adaptive = model.compat and model.compat.get("forceAdaptiveThinking")

            if is_adaptive:
                thinking: dict[str, Any] = {
                    "type": "adaptive",
                    "display": display,
                }
                params["thinking"] = thinking
                if options.effort:
                    # output_config 是顶层参数，不在 thinking 内
                    params["output_config"] = {
                        "effort": options.effort,
                    }
            else:
                # Budget-based
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": options.thinking_budget_tokens or 1024,
                    "display": display,
                }
        elif (
            options.thinking_enabled is False
            and model.thinking_level_map
            and model.thinking_level_map.get("off") is not None
        ):
            params["thinking"] = {"type": "disabled"}

    # Metadata
    if options and options.metadata:
        user_id = options.metadata.get("user_id")
        if isinstance(user_id, str):
            params["metadata"] = {"user_id": user_id}

    # Tool choice
    if options and options.tool_choice:
        if isinstance(options.tool_choice, str):
            params["tool_choice"] = {"type": options.tool_choice}
        else:
            params["tool_choice"] = options.tool_choice

    return params
