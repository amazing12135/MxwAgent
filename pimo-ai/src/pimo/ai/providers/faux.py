"""Faux（脚本化）Provider。

用于测试环境：不调用真实 LLM API，而是按预设脚本逐条消费预制的
AssistantMessage。支持 token 级别的流式模拟（分块 + 限速延迟），
以及基于文本长度的 token 用量估算（含会话级缓存命中模拟）。

使用示例::

    reg = register_faux_provider()
    reg.set_responses([
        faux_assistant_message("Hello! How can I help?")
    ])
    provider = FauxProvider(reg)
    stream = await provider.stream(model, context)
    async for event in stream:
        print(event)
    reg.unregister()
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import random as _random
import time
from dataclasses import dataclass, field
from typing import Any

from pimo.ai.api_registry import (
    ApiProvider,
    register_api_provider,
    unregister_api_providers,
)
from pimo.ai.event_stream import AssistantMessageEventStream
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
# 默认常量
# =============================================================================

_DEFAULT_API = "faux"
_DEFAULT_PROVIDER = "faux"
_DEFAULT_MODEL_ID = "faux-1"
_DEFAULT_MODEL_NAME = "Faux Model"
_DEFAULT_BASE_URL = "http://localhost:0"
_DEFAULT_MIN_TOKEN_SIZE = 3
_DEFAULT_MAX_TOKEN_SIZE = 5

_DEFAULT_USAGE = Usage(
    input=0,
    output=0,
    cache_read=0,
    cache_write=0,
    total_tokens=0,
    cost=CostInfo(),
)


def _random_id(prefix: str) -> str:
    """生成随机 ID。

    Args:
        prefix: ID 前缀。

    Returns:
        格式为 ``{prefix}:{timestamp}:{random}`` 的唯一 ID。
    """
    ts = int(time.time() * 1000)
    rnd = _random.randint(0, 0xFFFFFFFF).to_bytes(4, "big").hex()
    return f"{prefix}:{ts}:{rnd}"


# =============================================================================
# 类型定义
# =============================================================================


@dataclass(kw_only=True)
class FauxModelDefinition:
    """Faux Provider 使用的模型定义。

    简化版 Model，仅包含 Faux 需要的最小字段。
    """

    id: str
    name: str = ""
    reasoning: bool = False
    input: list[str] = field(default_factory=lambda: ["text", "image"])
    context_window: int = 128000
    max_tokens: int = 16384


FauxContentBlock = TextContent | ThinkingContent | ToolCall
"""Faux 支持的内容块类型。"""

# FauxResponseStep: 响应队列中的单个步骤
FauxResponseStep = AssistantMessage | Any
"""响应队列条目。可以是预制的 AssistantMessage 或动态工厂函数 (callable)。

工厂函数签名: (Context, StreamOptions | None, dict, Model) -> AssistantMessage | Awaitable[AssistantMessage]
"""


@dataclass
class TokenSizeOptions:
    """流式分块大小配置。"""

    min: int = 3
    """最小 chunk 的 token 数."""
    max: int = 5
    """最大 chunk 的 token 数."""


@dataclass
class RegisterFauxProviderOptions:
    """register_faux_provider() 的选项。"""

    api: str | None = None
    """API 协议名。None 时自动生成 ``faux:<random>``."""

    provider: str | None = None
    """Provider 名。默认 "faux"."""

    models: list[FauxModelDefinition] | None = None
    """模型定义列表。None 时使用默认 faux-1 模型。"""

    tokens_per_second: float | None = None
    """流式模拟速率（tokens/秒）。None 表示不延迟。"""

    token_size: TokenSizeOptions | None = None
    """Token 分块大小配置。"""


# =============================================================================
# FauxProviderRegistration — 注册后返回的控制句柄
# =============================================================================


class FauxProviderRegistration:
    """Faux Provider 注册句柄。

    提供模型查询、响应队列管理、调用状态追踪、以及反注册功能。
    """

    def __init__(
        self,
        *,
        api: str,
        source_id: str,
        models: list[Model],
        min_token_size: int,
        max_token_size: int,
        tokens_per_second: float | None,
    ) -> None:
        self.api = api
        self._source_id = source_id
        self.models = models
        self._min_token_size = min_token_size
        self._max_token_size = max_token_size
        self._tokens_per_second = tokens_per_second
        self._pending_responses: list[FauxResponseStep] = []
        self._prompt_cache: dict[str, str] = {}
        self._state = {"callCount": 0}

    # -------------------------------------------------------------------------
    # 公共接口
    # -------------------------------------------------------------------------

    def get_model(self, model_id: str | None = None) -> Model | None:
        """获取模型定义。

        Args:
            model_id: 模型 ID。为 None 时返回默认模型（列表第一个）。

        Returns:
            匹配的 Model，未找到返回 None。
        """
        if model_id is None:
            return self.models[0]
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def set_responses(self, responses: list[FauxResponseStep]) -> None:
        """替换响应队列（清空旧队列后填入新的响应序列）。

        Args:
            responses: 新的响应序列。
        """
        self._pending_responses = list(responses)

    def append_responses(self, responses: list[FauxResponseStep]) -> None:
        """追加响应到队列末尾。

        Args:
            responses: 要追加的响应序列。
        """
        self._pending_responses.extend(responses)

    def get_pending_response_count(self) -> int:
        """获取待消费的响应数量。

        Returns:
            队列中剩余的响应步骤数。
        """
        return len(self._pending_responses)

    def unregister(self) -> None:
        """反注册此 Provider。调用后该 Provider 不可再使用。"""
        unregister_api_providers(self._source_id)

    # -------------------------------------------------------------------------
    # 内部: 消费下一个响应步骤
    # -------------------------------------------------------------------------

    def _pop_response(self) -> FauxResponseStep | None:
        """从队列取出下一个响应步骤。"""
        if self._pending_responses:
            return self._pending_responses.pop(0)
        return None

    def _increment_call_count(self) -> None:
        self._state["callCount"] += 1


# =============================================================================
# FauxProvider
# =============================================================================


class FauxProvider(ApiProvider):
    """脚本化 Provider。

    不从网络调用 LLM，而是按预设队列消费 AssistantMessage。
    每条消息按 token 粒度分块推送，模拟真实流式行为。

    由 ``register_faux_provider()`` 创建并通过 ``FauxProviderRegistration``
    管理其响应队列和生命周期。
    """

    def __init__(self, registration: FauxProviderRegistration) -> None:
        """从注册句柄构造 Provider。

        Args:
            registration: register_faux_provider() 返回的控制句柄。
        """
        self.api = registration.api
        self._registration = registration

    # -------------------------------------------------------------------------
    # ApiProvider 接口实现
    # -------------------------------------------------------------------------

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """从响应队列中消费下一条消息，模拟流式推送。

        Args:
            model: 目标模型。
            context: 统一上下文。
            options: 流式选项。

        Returns:
            AssistantMessageEventStream: 模拟的异步事件流。
        """
        reg = self._registration
        stream = AssistantMessageEventStream()

        # 取队列中下一个响应步骤
        step = reg._pop_response()
        reg._increment_call_count()

        if step is None:
            error_msg = _create_error_message(
                RuntimeError("No more faux responses queued"),
                self.api,
                model.provider,
                model.id,
            )
            error_msg = _with_usage_estimate(
                error_msg, context, options, reg._prompt_cache,
            )
            stream.push(
                _make_error_event("error", error_msg)
            )
            stream.end(error_msg)
            return stream

        # 解析响应（支持工厂函数）
        if callable(step):
            try:
                result = step(context, options, reg._state, model)
                if asyncio.iscoroutine(result):
                    resolved = await result
                else:
                    resolved = result
            except Exception as exc:
                error_msg = _create_error_message(
                    exc, self.api, model.provider, model.id,
                )
                stream.push(
                    _make_error_event("error", error_msg)
                )
                stream.end(error_msg)
                return stream
        else:
            resolved = step

        # 克隆并标注 api/provider/model
        message = _clone_message(resolved, self.api, model.provider, model.id)
        message = _with_usage_estimate(
            message, context, options, reg._prompt_cache,
        )
        message.timestamp = message.timestamp or time.time()

        # 后台模拟流式推送
        asyncio.create_task(
            _stream_with_deltas(
                stream=stream,
                message=message,
                min_token_size=reg._min_token_size,
                max_token_size=reg._max_token_size,
                tokens_per_second=reg._tokens_per_second,
                signal=options.signal if options else None,
            )
        )
        return stream

    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """stream 的简化版本（直接委托 stream）。

        Args:
            model: 目标模型。
            context: 统一上下文。
            options: 简化选项。

        Returns:
            AssistantMessageEventStream: 模拟的异步事件流。
        """
        return await self.stream(model, context, options)


# =============================================================================
# 工厂函数 — 创建模拟数据
# =============================================================================


def faux_text(text: str) -> TextContent:
    """创建文本内容块。

    Args:
        text: 文本内容。

    Returns:
        TextContent 实例。
    """
    return TextContent(text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    """创建思考内容块。

    Args:
        thinking: 思考文本。

    Returns:
        ThinkingContent 实例。
    """
    return ThinkingContent(thinking=thinking)


def faux_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    id: str | None = None,
) -> ToolCall:
    """创建工具调用块。

    Args:
        name: 工具名。
        arguments: 调用参数。
        id: 工具调用 ID。None 时自动生成。

    Returns:
        ToolCall 实例。
    """
    return ToolCall(
        id=id or _random_id("tool"),
        name=name,
        arguments=arguments,
    )


def faux_assistant_message(
    content: str | FauxContentBlock | list[FauxContentBlock],
    *,
    stop_reason: str = "stop",
    error_message: str | None = None,
    response_id: str | None = None,
    timestamp: float | None = None,
) -> AssistantMessage:
    """创建完整的 AssistantMessage。

    Args:
        content: 消息内容。字符串自动转为 [TextContent]。
        stop_reason: 终止原因。默认 "stop"。
        error_message: 错误消息。
        response_id: 响应 ID。
        timestamp: 时间戳。None 时使用当前时间。

    Returns:
        可用于 Faux Provider 响应队列的 AssistantMessage。
    """
    if isinstance(content, str):
        normalized: list[FauxContentBlock] = [faux_text(content)]
    elif isinstance(content, list):
        normalized = content
    else:
        normalized = [content]

    return AssistantMessage(
        role="assistant",
        content=list(normalized),
        api=_DEFAULT_API,
        provider=_DEFAULT_PROVIDER,
        model=_DEFAULT_MODEL_ID,
        usage=_DEFAULT_USAGE,
        stop_reason=stop_reason,
        error_message=error_message,
        response_id=response_id,
        timestamp=timestamp or time.time(),
    )


# =============================================================================
# register_faux_provider() — 便利注册函数
# =============================================================================


def register_faux_provider(
    options: RegisterFauxProviderOptions | None = None,
) -> FauxProviderRegistration:
    """注册一个新的 Faux Provider。

    创建 Provider、注册到全局 Provider 注册表、返回控制句柄。

    Args:
        options: 注册选项。

    Returns:
        FauxProviderRegistration: 用于管理响应队列和生命周期的句柄。
    """
    opts = options or RegisterFauxProviderOptions()

    api = opts.api or _random_id(_DEFAULT_API)
    provider = opts.provider or _DEFAULT_PROVIDER
    source_id = _random_id("faux-provider")

    min_ts = opts.token_size.min if opts.token_size else _DEFAULT_MIN_TOKEN_SIZE
    max_ts = opts.token_size.max if opts.token_size else _DEFAULT_MAX_TOKEN_SIZE
    min_token_size = max(1, min(min_ts, max_ts))
    max_token_size = max(min_token_size, max_ts)
    tokens_per_second = opts.tokens_per_second

    # 模型定义
    model_defs = opts.models or [
        FauxModelDefinition(
            id=_DEFAULT_MODEL_ID,
            name=_DEFAULT_MODEL_NAME,
            reasoning=False,
            input=["text", "image"],
            context_window=128000,
            max_tokens=16384,
        )
    ]
    models: list[Model] = []
    for md in model_defs:
        cost = CostInfo()
        models.append(Model(
            id=md.id,
            name=md.name or md.id,
            api=api,
            provider=provider,
            base_url=_DEFAULT_BASE_URL,
            reasoning=md.reasoning,
            input_types=md.input,
            cost=cost,
            context_window=md.context_window,
            max_tokens=md.max_tokens,
        ))

    reg = FauxProviderRegistration(
        api=api,
        source_id=source_id,
        models=models,
        min_token_size=min_token_size,
        max_token_size=max_token_size,
        tokens_per_second=tokens_per_second,
    )

    # 创建 FauxProvider 实例并注册
    faux = FauxProvider(reg)
    register_api_provider(faux, source_id)

    return reg


# =============================================================================
# 内部辅助: 错误/克隆
# =============================================================================


def _create_error_message(
    error: Any,
    api: str,
    provider: str,
    model_id: str,
) -> AssistantMessage:
    """创建表示错误的 AssistantMessage。"""
    return AssistantMessage(
        role="assistant",
        content=[],
        api=api,
        provider=provider,
        model=model_id,
        usage=_DEFAULT_USAGE,
        stop_reason="error",
        error_message=str(error) if not isinstance(error, str) else error,
        timestamp=time.time(),
    )


def _create_aborted_message(partial: AssistantMessage) -> AssistantMessage:
    """标记消息为已取消。"""
    return AssistantMessage(
        role="assistant",
        content=partial.content,
        api=partial.api,
        provider=partial.provider,
        model=partial.model,
        usage=partial.usage,
        stop_reason="aborted",
        error_message="Request was aborted",
        timestamp=time.time(),
    )


def _clone_message(
    message: AssistantMessage,
    api: str,
    provider: str,
    model_id: str,
) -> AssistantMessage:
    """深拷贝 AssistantMessage 并替换 api/provider/model 字段。"""
    cloned = copy.deepcopy(message)
    cloned.api = api
    cloned.provider = provider
    cloned.model = model_id
    cloned.timestamp = cloned.timestamp or time.time()
    cloned.usage = cloned.usage if cloned.usage else _DEFAULT_USAGE
    return cloned


def _make_error_event(
    reason: str, message: AssistantMessage
) -> ErrorEvent:
    """创建错误事件。"""
    return ErrorEvent(message=message)


# =============================================================================
# Token 估算
# =============================================================================


def _estimate_tokens(text: str) -> int:
    """基于文本长度估算 token 数。

    使用 coarse 估算: 每 4 字符 ≈ 1 token。

    Args:
        text: 输入文本。

    Returns:
        估算的 token 数（向上取整）。
    """
    return math.ceil(len(text) / 4)


# =============================================================================
# 字符串分块
# =============================================================================


def _split_string_by_token_size(
    text: str, min_token_size: int, max_token_size: int
) -> list[str]:
    """将文本按随机 token 大小切分为 chunk 序列。

    每个 chunk 大小为 minTokenSize ~ maxTokenSize 个 token 之间的随机值，
    用于模拟自然的流式节奏。

    Args:
        text: 要切分的文本。
        min_token_size: 最小 chunk 的 token 数。
        max_token_size: 最大 chunk 的 token 数。

    Returns:
        chunk 字符串列表。
    """
    chunks: list[str] = []
    index = 0
    while index < len(text):
        ts = min_token_size + _random.randint(
            0, max_token_size - min_token_size
        )
        char_size = max(1, ts * 4)
        chunks.append(text[index:index + char_size])
        index += char_size
    return chunks if chunks else [""]


# =============================================================================
# 公共前缀
# =============================================================================


def _common_prefix_length(a: str, b: str) -> int:
    """计算两个字符串的公共前缀长度。

    用于缓存命中模拟：公共前缀部分视为 cache_read。

    Args:
        a: 第一个字符串。
        b: 第二个字符串。

    Returns:
        公共前缀的字符数。
    """
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


# =============================================================================
# Context 序列化
# =============================================================================


def _content_to_text(content: str | list[TextContent | ImageContent]) -> str:
    """将 user content 转为文本摘要。"""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if hasattr(block, "type") and block.type == "text":
            parts.append(block.text)
        elif hasattr(block, "type") and block.type == "image":
            parts.append(f"[image:{block.mime_type}:{len(block.data)}]")
    return "\n".join(parts)


def _assistant_content_to_text(
    content: list[TextContent | ThinkingContent | ToolCall],
) -> str:
    """将 assistant content 转为文本摘要。"""
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ThinkingContent):
            parts.append(block.thinking)
        elif isinstance(block, ToolCall):
            parts.append(
                f"{block.name}:{json.dumps(block.arguments)}"
            )
    return "\n".join(parts)


def _tool_result_to_text(message: ToolResultMessage) -> str:
    """将工具结果转为文本摘要。"""
    parts = [message.tool_name]
    for block in message.content:
        parts.append(_content_to_text([block]))
    return "\n".join(parts)


def _message_to_text(message: Message) -> str:
    """将任一消息转为文本摘要。"""
    if message.role == "user":
        return _content_to_text(message.content)
    if message.role == "assistant":
        return _assistant_content_to_text(message.content)
    return _tool_result_to_text(message)


def _serialize_context(context: Context) -> str:
    """将 Context 序列化为文本（用于 token 估算）。"""
    parts: list[str] = []
    if context.system_prompt:
        parts.append(f"system:{context.system_prompt}")
    for msg in context.messages:
        parts.append(
            f"{msg.role}:{_message_to_text(msg)}"
        )
    if context.tools and len(context.tools) > 0:
        parts.append(f"tools:{json.dumps(context.tools)}")
    return "\n\n".join(parts)


# =============================================================================
# 用量估算（含缓存模拟）
# =============================================================================


def _with_usage_estimate(
    message: AssistantMessage,
    context: Context,
    options: StreamOptions | None,
    prompt_cache: dict[str, str],
) -> AssistantMessage:
    """为消息附加估算的 token 用量。

    基于上下文和回复文本长度估算 input/output/cacheRead/cacheWrite 的
    token 数。支持基于 session_id 的缓存命中模拟。

    Args:
        message: 要附加用量的消息。
        context: 请求上下文。
        options: 流式选项（取 sessionId、cacheRetention）。
        prompt_cache: 会话级缓存字典。

    Returns:
        已附加用量的 AssistantMessage（原地修改）。
    """
    prompt_text = _serialize_context(context)
    prompt_tokens = _estimate_tokens(prompt_text)
    output_tokens = _estimate_tokens(
        _assistant_content_to_text(message.content)
    )

    input_tokens = prompt_tokens
    cache_read = 0
    cache_write = 0
    session_id = options.session_id if options else None
    cache_retention = options.cache_retention if options else None

    if session_id and cache_retention != "none":
        previous = prompt_cache.get(session_id)
        if previous:
            cached_chars = _common_prefix_length(previous, prompt_text)
            cache_read = _estimate_tokens(previous[:cached_chars])
            cache_write = _estimate_tokens(
                prompt_text[cached_chars:]
            )
            input_tokens = max(0, prompt_tokens - cache_read)
        else:
            cache_write = prompt_tokens
        prompt_cache[session_id] = prompt_text

    message.usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input_tokens + output_tokens + cache_read + cache_write,
        cost=CostInfo(),
    )
    return message


# =============================================================================
# 流式延迟
# =============================================================================


async def _schedule_chunk(
    chunk: str, tokens_per_second: float | None
) -> None:
    """按 token 速率延迟。

    tokens_per_second 为 None 时立即 yield 控制权 (asyncio.sleep(0))。

    Args:
        chunk: 当前 chunk 文本。
        tokens_per_second: 目标速率（tokens/秒）。
    """
    if not tokens_per_second or tokens_per_second <= 0:
        await asyncio.sleep(0)
        return

    delay_ms = (_estimate_tokens(chunk) / tokens_per_second) * 1000
    await asyncio.sleep(delay_ms / 1000)


# =============================================================================
# 核心: 流式模拟
# =============================================================================


async def _stream_with_deltas(
    stream: AssistantMessageEventStream,
    message: AssistantMessage,
    min_token_size: int,
    max_token_size: int,
    tokens_per_second: float | None,
    signal: Any,
) -> None:
    """将 AssistantMessage 模拟为流式事件序列。

    对 message.content 中的每个块:
    - thinking → 按 chunk 推送 start → delta(+) → done
    - text → 按 chunk 推送 start → delta(+) → done
    - toolCall → 按 chunk 推送 start → delta(+) → done

    Args:
        stream: 事件流。
        message: 要模拟的完整消息。
        min_token_size: 最小 chunk 的 token 数。
        max_token_size: 最大 chunk 的 token 数。
        tokens_per_second: 流式速率。
        signal: 取消信号。
    """
    def _is_aborted() -> bool:
        return signal and getattr(signal, "is_set", lambda: False)()

    try:
        # 构建 partial 消息（content 逐步追加）
        partial = copy.deepcopy(message)
        partial.content = []

        if _is_aborted():
            aborted = _create_aborted_message(partial)
            stream.push(_make_error_event("aborted", aborted))
            stream.end(aborted)
            return

        # 推送 start
        stream.push(StartEvent(message=copy.deepcopy(partial)))

        for block in message.content:
            if _is_aborted():
                aborted = _create_aborted_message(partial)
                stream.push(_make_error_event("aborted", aborted))
                stream.end(aborted)
                return

            if isinstance(block, ThinkingContent):
                # 追加空 thinking 块到 partial
                thinking_partial = copy.deepcopy(partial)
                thinking_partial.content.append(
                    ThinkingContent(thinking="", signature=block.signature)
                )
                partial = thinking_partial

                # 分块推送
                for chunk in _split_string_by_token_size(
                    block.thinking, min_token_size, max_token_size
                ):
                    await _schedule_chunk(chunk, tokens_per_second)
                    if _is_aborted():
                        aborted = _create_aborted_message(partial)
                        stream.push(_make_error_event("aborted", aborted))
                        stream.end(aborted)
                        return
                    partial.content[-1].thinking += chunk
                    stream.push(ThinkingDeltaEvent(message=copy.deepcopy(partial)))

            elif isinstance(block, TextContent):
                # 追加空 text 块到 partial
                text_partial = copy.deepcopy(partial)
                text_partial.content.append(TextContent(text=""))
                partial = text_partial

                for chunk in _split_string_by_token_size(
                    block.text, min_token_size, max_token_size
                ):
                    await _schedule_chunk(chunk, tokens_per_second)
                    if _is_aborted():
                        aborted = _create_aborted_message(partial)
                        stream.push(_make_error_event("aborted", aborted))
                        stream.end(aborted)
                        return
                    partial.content[-1].text += chunk
                    stream.push(TextDeltaEvent(message=copy.deepcopy(partial)))

            elif isinstance(block, ToolCall):
                # 追加空 toolCall 到 partial（arguments 渐进填充）
                tc_partial = copy.deepcopy(partial)
                tc_partial.content.append(
                    ToolCall(id=block.id, name=block.name, arguments={})
                )
                partial = tc_partial

                args_json = json.dumps(block.arguments)
                for chunk in _split_string_by_token_size(
                    args_json, min_token_size, max_token_size
                ):
                    await _schedule_chunk(chunk, tokens_per_second)
                    if _is_aborted():
                        aborted = _create_aborted_message(partial)
                        stream.push(_make_error_event("aborted", aborted))
                        stream.end(aborted)
                        return
                    stream.push(ToolCallDeltaEvent(message=copy.deepcopy(partial)))

                # 最终填入完整 arguments
                partial.content[-1].arguments = block.arguments

        # 推送终止事件
        if message.stop_reason in ("error", "aborted"):
            stream.push(_make_error_event(message.stop_reason, message))
        else:
            stream.push(DoneEvent(message=message))
        stream.end(message)

    except Exception as exc:
        error_msg = _create_error_message(
            exc, message.api, message.provider, message.model,
        )
        stream.push(ErrorEvent(message=error_msg))
        stream.end(error_msg)
