"""OpenAI Responses 协议 Provider。

通过 OpenAI Python SDK 调用 OpenAI Responses API，将厂商原生 SSE 流
归一化为 pimo-ai 统一事件流。

Responses API 是 OpenAI 的新一代 API，相比 Chat Completions 提供:
- 原生 reasoning/thinking 支持 (response.reasoning_* 事件)
- service_tier 定价 (flex/priority/default)
- 结构化 output_item 生命周期 (added → delta → done)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from pimo.ai.api_registry import ApiProvider
from pimo.ai.event_stream import AssistantMessageEventStream
from pimo.ai.models.registry import clamp_thinking_level
from pimo.ai.types import (
    AssistantMessage,
    Context,
    CostInfo,
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    Usage,
)

from pimo.ai.providers.openai_responses_shared import (
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)

# OpenAI Responses 协议允许 tool call ID 直传的 provider 集合
_OPENAI_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}

# =============================================================================
# OpenAI Responses 特有类型
# =============================================================================


@dataclass(kw_only=True)
class OpenAIResponsesOptions(StreamOptions):
    """OpenAI Responses API 扩展选项。

    在 StreamOptions 基础上增加 reasoning、service_tier 等参数。
    """

    reasoning_effort: str | None = None
    """思考/推理努力级别。映射自 pimo ThinkingLevel。
    None 表示不启用 reasoning。
    """

    reasoning_summary: str | None = None
    """推理摘要模式: ``"auto"`` | ``"detailed"`` | ``"concise"`` | ``None``。
    控制 reasoning summary 的详细程度。
    """

    service_tier: str | None = None
    """服务层级: ``"auto"`` | ``"default"`` | ``"flex"`` | ``"priority"``。
    影响价格和响应速度。
    """


# =============================================================================
# OpenAIResponsesProvider
# =============================================================================


class OpenAIResponsesProvider(ApiProvider):
    """OpenAI Responses API Provider。

    封装 OpenAI Python SDK 的 Responses API，支持 reasoning、service_tier
    和新一代事件协议。流处理委托给 ``process_responses_stream()``。
    """

    api: str = "openai-responses"

    # -------------------------------------------------------------------------
    # ApiProvider 接口实现
    # -------------------------------------------------------------------------

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """流式调用 OpenAI Responses API。

        Args:
            model: 目标模型描述（api="openai-responses"）。
            context: 统一上下文。
            options: 流式选项。

        Returns:
            AssistantMessageEventStream: 归一化后的异步事件流。
        """
        stream = AssistantMessageEventStream()
        api_key = options.api_key if options else None
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        compat = _get_compat(model)
        cache_retention = _resolve_cache_retention(
            options.cache_retention if options else None
        )
        session_id = (
            None
            if cache_retention == "none"
            else (options.session_id if options else None)
        )
        client = _create_client(
            model=model,
            api_key=api_key,
            compat=compat,
            options_headers=options.headers if options else None,
            session_id=session_id,
        )

        asyncio.create_task(
            self._run_stream(
                model=model,
                context=context,
                options=options,
                stream=stream,
                compat=compat,
                cache_retention=cache_retention,
                client=client,
            )
        )
        return stream

    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """stream 的简化版本，额外支持 reasoning 参数。

        - reasoning=None 或 "off": 不传 reasoning_effort
        - 其他: 将 ThinkingLevel clamp 后作为 reasoning_effort 传入

        Args:
            model: 目标模型描述。
            context: 统一上下文。
            options: 包含 reasoning 的简化选项。

        Returns:
            AssistantMessageEventStream: 归一化后的异步事件流。
        """
        api_key = options.api_key if options else None
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        reasoning = options.reasoning if options else None
        clamped = clamp_thinking_level(model, reasoning) if reasoning else None
        reasoning_effort = None if clamped == "off" else clamped

        merged = OpenAIResponsesOptions(
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
            reasoning_effort=reasoning_effort,
        )

        return await self.stream(model, context, merged)

    # -------------------------------------------------------------------------
    # 核心流式执行
    # -------------------------------------------------------------------------

    async def _run_stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None,
        stream: AssistantMessageEventStream,
        compat: dict[str, Any],
        cache_retention: str,
        client: Any = None,
    ) -> None:
        """在后台异步执行流式调用，将事件推入 stream。

        Args:
            model: 目标模型。
            context: 统一上下文。
            options: 流式选项。
            stream: 事件流。
            compat: 已解析的兼容性配置。
            cache_retention: 缓存保留策略。
            client: 预构建的 AsyncOpenAI 客户端实例。
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
                OpenAIResponsesOptions(**options.__dict__)
                if options and not isinstance(options, OpenAIResponsesOptions)
                else options
            ) if options else None

            # 1. 构建请求参数
            params = _build_params(model, context, opts, compat, cache_retention)

            # 2. onPayload 回调
            if opts and opts.on_payload:
                opts.on_payload(params)

            # 3. 发起流式请求
            sdk_stream = await client.responses.create(**params)

            # 4. 推送 start 事件
            stream.push(StartEvent(message=output))

            # 5. 委托共享模块处理 Responses 事件流
            stream_options: dict[str, Any] = {}
            if opts and opts.service_tier:
                stream_options["service_tier"] = opts.service_tier
                stream_options["apply_service_tier_pricing"] = (
                    lambda usage, tier: _apply_service_tier_pricing(usage, tier, model)
                )
                stream_options["resolve_service_tier"] = (
                    lambda response_tier, request_tier: response_tier or request_tier
                )

            await process_responses_stream(
                sdk_stream, output, stream, model,
                stream_options if stream_options else None,
            )

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
                if hasattr(block, "_partial_json"):
                    delattr(block, "_partial_json")
                if hasattr(block, "_index"):
                    delattr(block, "_index")

            is_aborted = (
                opts
                and opts.signal
                and getattr(opts.signal, "is_set", lambda: False)()
            )
            output.stop_reason = "aborted" if is_aborted else "error"
            output.error_message = _format_responses_error(exc)

            stream.push(ErrorEvent(message=output))


def _format_responses_error(error: Any) -> str:
    """格式化 OpenAI Responses 错误消息。

    Args:
        error: 异常对象。

    Returns:
        可读的错误描述字符串。
    """
    if isinstance(error, Exception):
        status = getattr(error, "status", None)
        if isinstance(status, int):
            return f"OpenAI API error ({status}): {error}"
        return str(error)
    try:
        return json.dumps(error)
    except Exception:
        return str(error)


# =============================================================================
# 兼容性
# =============================================================================


def _get_compat(model: Model) -> dict[str, Any]:
    """获取模型的 OpenAI Responses 兼容性配置。

    model.compat 显式设置优先，否则使用默认值。

    Args:
        model: 目标模型。

    Returns:
        完整解析的 compat dict。
    """
    mc = model.compat or {}
    return {
        "supportsDeveloperRole": mc.get("supportsDeveloperRole", True),
        "sendSessionIdHeader": mc.get("sendSessionIdHeader", True),
        "supportsLongCacheRetention": mc.get(
            "supportsLongCacheRetention", True
        ),
    }


# =============================================================================
# 缓存保留
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


def _get_prompt_cache_retention(
    compat: dict[str, Any], cache_retention: str
) -> str | None:
    """计算 prompt_cache_retention 值。

    Args:
        compat: 兼容性配置。
        cache_retention: 缓存保留策略。

    Returns:
        "24h" 或 None。
    """
    if cache_retention == "long" and compat.get(
        "supportsLongCacheRetention", True
    ):
        return "24h"
    return None


# =============================================================================
# 请求参数构建
# =============================================================================


def _build_params(
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions | None,
    compat: dict[str, Any],
    cache_retention: str,
) -> dict[str, Any]:
    """构建 OpenAI Responses API 请求参数。

    Args:
        model: 目标模型。
        context: 统一上下文。
        options: OpenAI Responses 扩展选项。
        compat: 兼容性配置。
        cache_retention: 缓存保留策略。

    Returns:
        OpenAI ResponseCreateParams 兼容的参数字典。
    """
    messages = convert_responses_messages(
        model, context, _OPENAI_TOOL_CALL_PROVIDERS,
    )

    params: dict[str, Any] = {
        "model": model.id,
        "input": messages,
        "stream": True,
        "store": False,
    }

    # prompt_cache_key
    if cache_retention != "none":
        session_id = options.session_id if options else None
        if session_id:
            params["prompt_cache_key"] = session_id[:64]

    # prompt_cache_retention
    retention = _get_prompt_cache_retention(compat, cache_retention)
    if retention:
        params["prompt_cache_retention"] = retention

    # max_output_tokens
    if options and options.max_tokens is not None:
        params["max_output_tokens"] = options.max_tokens

    # temperature
    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    # service_tier
    if options and options.service_tier is not None:
        params["service_tier"] = options.service_tier

    # tools
    if context.tools and len(context.tools) > 0:
        params["tools"] = convert_responses_tools(context.tools)

    # reasoning
    if model.reasoning:
        if options and (
            options.reasoning_effort or options.reasoning_summary
        ):
            effort = (
                model.thinking_level_map.get(options.reasoning_effort)
                if (
                    options
                    and options.reasoning_effort
                    and model.thinking_level_map
                )
                else (options.reasoning_effort if options else None)
            )
            if not effort:
                effort = "medium"
            params["reasoning"] = {
                "effort": effort,
                "summary": (
                    options.reasoning_summary
                    if (options and options.reasoning_summary)
                    else "auto"
                ),
            }
            params["include"] = ["reasoning.encrypted_content"]
        elif (
            model.thinking_level_map
            and model.thinking_level_map.get("off") is not None
        ):
            params["reasoning"] = {
                "effort": model.thinking_level_map["off"],
            }

    return params


# =============================================================================
# Service Tier 定价
# =============================================================================


def _get_service_tier_cost_multiplier(
    model: Model,
    service_tier: str | None,
) -> float:
    """根据 service tier 获取价格倍率。

    flex=0.5, priority=2 (或 gpt-5.5 时为 2.5), 默认=1

    Args:
        model: 目标模型。
        service_tier: 服务层级。

    Returns:
        价格倍率。
    """
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        return 2.5 if model.id == "gpt-5.5" else 2.0
    return 1.0


def _apply_service_tier_pricing(
    usage: Usage,
    service_tier: str | None,
    model: Model,
) -> None:
    """将 service tier 价格倍率应用到 usage.cost 各字段。

    Args:
        usage: Usage 对象（原地修改 cost）。
        service_tier: 服务层级。
        model: 目标模型。
    """
    multiplier = _get_service_tier_cost_multiplier(model, service_tier)
    if multiplier == 1.0:
        return

    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cache_read *= multiplier
    usage.cost.cache_write *= multiplier
    usage.cost.total = (
        usage.cost.input
        + usage.cost.output
        + usage.cost.cache_read
        + usage.cost.cache_write
    )


# =============================================================================
# SDK 客户端构造
# =============================================================================


def _create_client(
    model: Model,
    api_key: str,
    compat: dict[str, Any],
    *,
    options_headers: dict[str, str] | None = None,
    session_id: str | None = None,
) -> AsyncOpenAI:
    """构造 OpenAI SDK 客户端（用于 Responses API）。

    根据 compat 自动配置:
    - session_id header（用于缓存亲和）
    - x-client-request-id header

    Args:
        model: 目标模型。
        api_key: API Key。
        compat: 兼容性配置。
        options_headers: 调用方附加 HTTP 头。
        session_id: 缓存亲和会话 ID。

    Returns:
        配置完成的 OpenAI SDK 客户端实例。
    """
    headers: dict[str, str] = {}

    # Merge model-level headers (from model.compat header: prefix)
    if model.compat and isinstance(model.compat, dict):
        model_headers = {
            k[len("header:"):]: str(v)
            for k, v in model.compat.items()
            if k.startswith("header:")
        }
        headers.update(model_headers)

    if session_id:
        if compat.get("sendSessionIdHeader", True):
            headers["session_id"] = session_id
        headers["x-client-request-id"] = session_id

    if options_headers:
        headers.update(options_headers)

    return AsyncOpenAI(
        api_key=api_key,
        base_url=model.base_url,
        default_headers=headers if headers else None,
        max_retries=0,
    )
