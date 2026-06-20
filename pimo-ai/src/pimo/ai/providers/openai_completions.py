"""OpenAI Completions 协议 Provider。

通过 OpenAI Python SDK 调用 OpenAI Chat Completions API（及兼容厂商），
将厂商原生 SSE 流归一化为 pimo-ai 统一事件流。

覆盖 80%+ LLM 厂商：OpenAI、DeepSeek、OpenRouter、Groq、Together、xAI 等。

首批支持的 thinking 格式:
- ``"openai"`` — 标准 reasoning_effort
- ``"deepseek"`` — thinking: {type: enabled/disabled}
- ``"openrouter"`` — reasoning: {effort}
"""

from __future__ import annotations
import json
import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from openai import AsyncOpenAI

from pimo.ai.api_registry import ApiProvider
from pimo.ai.event_stream import AssistantMessageEventStream
from pimo.ai.models.cost import calculate_cost
from pimo.ai.models.registry import clamp_thinking_level
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
# OpenAI Completions 特有类型
# =============================================================================


@dataclass(kw_only=True)
class OpenAICompletionsOptions(StreamOptions):
    """OpenAI Chat Completions API 扩展选项。

    在 StreamOptions 基础上增加 tool_choice 和 reasoning_effort 参数。
    """

    tool_choice: str | dict[str, Any] | None = None
    """工具选择行为。

    - ``"auto"`` | ``"none"`` | ``"required"``: OpenAI 内置选择
    - ``{"type": "function", "function": {"name": "..."}}``: 强制指定函数
    - ``None``: 省略
    """

    reasoning_effort: str | None = None
    """思考/推理努力级别。映射自 pimo ThinkingLevel。
    None 表示不启用 reasoning。
    """


# =============================================================================
# OpenAICompletionsProvider
# =============================================================================


class OpenAICompletionsProvider(ApiProvider):
    """OpenAI Chat Completions API Provider。

    封装 OpenAI Python SDK，支持标准 API Key 和所有 OpenAI 兼容厂商。
    通过 SDK 内置 streaming 迭代 chunk，逐 chunk 转换为归一化事件。

    兼容性配置 (compat) 自动检测 provider + baseUrl 特征:
    - 标准 OpenAI: reasoning_effort, max_completion_tokens, store=False
    - DeepSeek: thinking {type}, reasoning_effort, max_tokens
    - OpenRouter: reasoning {effort}, anthropic cache_control
    """

    api: str = "openai-completions"

    # -------------------------------------------------------------------------
    # ApiProvider 接口实现
    # -------------------------------------------------------------------------

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """流式调用 OpenAI Chat Completions API。

        Args:
            model: 目标模型描述（api="openai-completions"）。
            context: 统一上下文（system_prompt + messages + tools）。
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

        merged = OpenAICompletionsOptions(
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
        cache_retention: str = "short",
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
                OpenAICompletionsOptions(**options.__dict__)
                if options and not isinstance(options, OpenAICompletionsOptions)
                else options
            ) if options else None

            # 1. 构建请求参数
            params = _build_params(model, context, opts, compat, cache_retention)

            # 2. onPayload 回调
            if opts and opts.on_payload:
                opts.on_payload(params)

            # 3. 发起流式请求
            sdk_stream = await client.chat.completions.create(**params)

            # 4. 推送 start 事件
            stream.push(StartEvent(message=output))

            # ---- content block trackers (单 text/thinking + 多 toolCall) ----
            text_block: TextContent | None = None
            thinking_block: ThinkingContent | None = None
            tool_blocks_by_index: dict[int, dict[str, Any]] = {}
            tool_blocks_by_id: dict[str, dict[str, Any]] = {}
            has_finish_reason = False

            def _get_content_index(block: Any) -> int:
                for i, b in enumerate(output.content):
                    if b is block:
                        return i
                return -1

            def _ensure_text_block() -> TextContent:
                nonlocal text_block
                if text_block is None:
                    text_block = TextContent(text="")
                    output.content.append(text_block)
                return text_block

            def _ensure_thinking_block(signature: str) -> ThinkingContent:
                nonlocal thinking_block
                if thinking_block is None:
                    thinking_block = ThinkingContent(
                        thinking="",
                        signature=signature,
                    )
                    output.content.append(thinking_block)
                return thinking_block

            # 5. 迭代 chunk
            async for chunk in sdk_stream:
                if opts and opts.signal:
                    if getattr(opts.signal, "is_set", lambda: False)():
                        raise asyncio.CancelledError("Request was aborted")

                if not chunk or not hasattr(chunk, "choices"):
                    continue

                # response id / model
                chunk_id = getattr(chunk, "id", None)
                if chunk_id and not output.response_id:
                    output.response_id = chunk_id
                chunk_model = getattr(chunk, "model", None)
                if (
                    isinstance(chunk_model, str)
                    and chunk_model
                    and chunk_model != model.id
                    and not getattr(output, "response_model", None)
                ):
                    output.response_model = chunk_model

                # usage (在含 usage 的 chunk 中直接替换)
                if getattr(chunk, "usage", None):
                    new_usage = _parse_chunk_usage(
                        {"usage_obj": chunk.usage}, model
                    )
                    output.usage.input = new_usage["input"]
                    output.usage.output = new_usage["output"]
                    output.usage.cache_read = new_usage["cache_read"]
                    output.usage.cache_write = new_usage["cache_write"]
                    output.usage.total_tokens = new_usage["total_tokens"]
                    output.usage.cost = new_usage["cost"]

                choices = getattr(chunk, "choices", None)
                if not choices or not isinstance(choices, list) or len(choices) == 0:
                    continue
                choice = choices[0]

                # fallback: usage in choice (Moonshot)
                if not getattr(chunk, "usage", None):
                    choice_usage = getattr(choice, "usage", None)
                    if choice_usage:
                        new_usage = _parse_chunk_usage(
                            {"usage_obj": choice_usage}, model
                        )
                        output.usage.input = new_usage["input"]
                        output.usage.output = new_usage["output"]
                        output.usage.cache_read = new_usage["cache_read"]
                        output.usage.cache_write = new_usage["cache_write"]
                        output.usage.total_tokens = new_usage["total_tokens"]
                        output.usage.cost = new_usage["cost"]

                # finish_reason
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason:
                    sr, err = _map_stop_reason(finish_reason)
                    output.stop_reason = sr
                    if err:
                        output.error_message = err
                    has_finish_reason = True

                # delta
                delta = getattr(choice, "delta", None)
                if not delta:
                    continue

                # ---- text delta ----
                delta_content = getattr(delta, "content", None)
                if delta_content:
                    block = _ensure_text_block()
                    block.text += delta_content
                    stream.push(TextDeltaEvent(message=output))

                # ---- thinking/reasoning delta ----
                reasoning_fields = ["reasoning_content", "reasoning", "reasoning_text"]
                found_field: str | None = None
                for field in reasoning_fields:
                    val = getattr(delta, field, None)
                    if isinstance(val, str) and val:
                        found_field = field
                        break

                if found_field:
                    signature = (
                        "reasoning_content"
                        if (
                            model.provider == "opencode-go"
                            and found_field == "reasoning"
                        )
                        else found_field
                    )
                    block = _ensure_thinking_block(signature)
                    delta_value = getattr(delta, found_field)
                    block.thinking += delta_value
                    stream.push(ThinkingDeltaEvent(message=output))

                # ---- tool call deltas ----
                tool_calls = getattr(delta, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        tc_index = getattr(tc, "index", None)
                        tc_id = getattr(tc, "id", None)
                        tc_func = getattr(tc, "function", None)

                        # find or create tool block
                        block_dict: dict[str, Any] | None = None
                        if tc_index is not None and tc_index in tool_blocks_by_index:
                            block_dict = tool_blocks_by_index[tc_index]
                        elif tc_id and tc_id in tool_blocks_by_id:
                            block_dict = tool_blocks_by_id[tc_id]

                        if block_dict is None:
                            block_dict = {
                                "_obj": ToolCall(
                                    id=tc_id or "",
                                    name=tc_func.name if tc_func else "",
                                    arguments={},
                                ),
                                "_partial_json": "",
                                "_stream_index": tc_index,
                            }
                            if tc_index is not None:
                                tool_blocks_by_index[tc_index] = block_dict
                            if tc_id:
                                tool_blocks_by_id[tc_id] = block_dict
                            output.content.append(block_dict["_obj"])

                        block_obj: ToolCall = block_dict["_obj"]

                        if tc_id and not block_obj.id:
                            block_obj.id = tc_id
                            tool_blocks_by_id[tc_id] = block_dict
                        if tc_func and tc_func.name and not block_obj.name:
                            block_obj.name = tc_func.name

                        delta_args = tc_func.arguments if tc_func else ""
                        if delta_args:
                            block_dict["_partial_json"] += delta_args
                            # progressive parse (best-effort)
                            try:
                                block_obj.arguments = json.loads(
                                    block_dict["_partial_json"]
                                )
                            except json.JSONDecodeError:
                                pass

                        stream.push(ToolCallDeltaEvent(message=output))

            # 6. Finish all blocks
            _finish_block(text_block, output, stream)
            _finish_block(thinking_block, output, stream)
            for bd in list(tool_blocks_by_index.values()):
                tc_obj = bd["_obj"]
                try:
                    tc_obj.arguments = json.loads(bd["_partial_json"])
                except (json.JSONDecodeError, KeyError):
                    pass
                # clean scratch fields
            for bd in list(tool_blocks_by_index.values()) + list(
                tool_blocks_by_id.values()
            ):
                bd.pop("_partial_json", None)
                bd.pop("_stream_index", None)

            # 7. 检查取消
            if opts and opts.signal:
                if getattr(opts.signal, "is_set", lambda: False)():
                    raise asyncio.CancelledError("Request was aborted")

            # 8. 检查终止原因
            if output.stop_reason == "aborted":
                raise RuntimeError("Request was aborted")
            if output.stop_reason == "error":
                raise RuntimeError(
                    output.error_message or "Provider returned an error stop reason"
                )
            if not has_finish_reason:
                raise RuntimeError("Stream ended without finish_reason")

            # 9. 推送完成事件
            stream.push(DoneEvent(message=output))

        except (asyncio.CancelledError, Exception) as exc:
            # 清理临时字段
            for block in output.content:
                if hasattr(block, "_partial_json"):
                    delattr(block, "_partial_json")
                if hasattr(block, "_stream_index"):
                    delattr(block, "_stream_index")

            is_aborted = (
                opts
                and opts.signal
                and getattr(opts.signal, "is_set", lambda: False)()
            )
            output.stop_reason = "aborted" if is_aborted else "error"
            output.error_message = str(exc)
            if isinstance(exc, asyncio.CancelledError):
                output.error_message = "Request was aborted"

            stream.push(ErrorEvent(message=output))


def _finish_block(
    block: TextContent | ThinkingContent | None,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
) -> None:
    """推送块最终状态（对齐 pi 的 finishBlock）。"""
    if block is None:
        return
    if isinstance(block, TextContent):
        stream.push(TextDeltaEvent(message=output))
    elif isinstance(block, ThinkingContent):
        stream.push(ThinkingDeltaEvent(message=output))


# =============================================================================
# 公共转换函数（被其他 OpenAI 系列 Provider 复用）
# =============================================================================


def convert_messages(
    model: Model,
    context: Context,
    compat: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """将统一 Message 列表转换为 OpenAI Chat Completions 消息格式。

    被 ``openai_responses.py``、``openai_codex_responses.py`` 等模块复用。

    Args:
        model: 目标模型。
        context: 统一上下文。
        compat: 兼容性配置。None 时自动检测。

    Returns:
        OpenAI ChatCompletionMessageParam 列表。
    """
    if compat is None:
        compat = _get_compat(model)

    params: list[dict[str, Any]] = []

    # System prompt
    if context.system_prompt:
        use_developer = model.reasoning and compat.get("supportsDeveloperRole", False)
        role = "developer" if use_developer else "system"
        params.append({"role": role, "content": context.system_prompt})

    last_role: str | None = None
    i = 0

    while i < len(context.messages):
        msg = context.messages[i]

        # Bridging assistant message between tool result and user
        if (
            compat.get("requiresAssistantAfterToolResult")
            and last_role == "tool"
            and msg.role == "user"
        ):
            params.append({
                "role": "assistant",
                "content": "I have processed the tool results.",
            })

        if msg.role == "user":
            if isinstance(msg.content, str):
                params.append({"role": "user", "content": msg.content})
            elif isinstance(msg.content, list):
                text_parts = [
                    c.text
                    for c in msg.content
                    if c.type == "text" and c.text.strip()
                ]
                has_images = any(c.type == "image" for c in msg.content)

                if not has_images and text_parts:
                    params.append({
                        "role": "user",
                        "content": "\n".join(text_parts),
                    })
                elif has_images or text_parts:
                    content_parts: list[dict[str, Any]] = []
                    for c in msg.content:
                        if c.type == "text" and c.text.strip():
                            content_parts.append({
                                "type": "text",
                                "text": c.text,
                            })
                        elif c.type == "image":
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{c.mime_type};base64,{c.data}",
                                },
                            })
                    if content_parts:
                        params.append({
                            "role": "user",
                            "content": content_parts,
                        })

            last_role = "user"

        elif msg.role == "assistant":
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": (
                    ""
                    if compat.get("requiresAssistantAfterToolResult")
                    else None
                ),
            }

            text_parts = [
                c.text
                for c in msg.content
                if isinstance(c, TextContent) and c.text.strip()
            ]
            assistant_text = "".join(text_parts)

            thinking_blocks = [
                c
                for c in msg.content
                if isinstance(c, ThinkingContent)
                and c.thinking.strip()
            ]

            if thinking_blocks:
                if compat.get("requiresThinkingAsText"):
                    thinking_text = "\n\n".join(
                        tb.thinking for tb in thinking_blocks
                    )
                    content_array: list[dict[str, Any]] = [
                        {"type": "text", "text": thinking_text}
                    ]
                    if assistant_text:
                        content_array.append({
                            "type": "text",
                            "text": assistant_text,
                        })
                    assistant_msg["content"] = content_array
                else:
                    if assistant_text:
                        assistant_msg["content"] = assistant_text
                    # 注入思考签名字段
                    sig = thinking_blocks[0].signature
                    if (
                        sig
                        and sig.strip()
                        and model.provider == "opencode-go"
                        and sig == "reasoning"
                    ):
                        sig = "reasoning_content"
                    if sig and sig.strip():
                        assistant_msg[sig] = "\n".join(
                            tb.thinking for tb in thinking_blocks
                        )
            elif assistant_text:
                assistant_msg["content"] = assistant_text

            # Tool calls
            tool_calls = [
                c for c in msg.content if isinstance(c, ToolCall)
            ]
            if tool_calls:
                assistant_msg["tool_calls"] = []
                for tc in tool_calls:
                    assistant_msg["tool_calls"].append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    })

            # Skip empty assistant messages (no content, no tool_calls)
            content = assistant_msg.get("content")
            has_content = (
                content is not None
                and (isinstance(content, str) and len(content) > 0
                     or isinstance(content, list) and len(content) > 0)
            )
            if not has_content and not assistant_msg.get("tool_calls"):
                i += 1
                continue

            params.append(assistant_msg)
            last_role = "assistant"

        elif msg.role == "toolResult":
            image_blocks: list[dict[str, Any]] = []
            j = i

            while (
                j < len(context.messages)
                and context.messages[j].role == "toolResult"
            ):
                tr = context.messages[j]
                text_result = "\n".join(
                    c.text
                    for c in tr.content
                    if isinstance(c, TextContent)
                )
                has_tool_images = any(
                    c.type == "image" for c in tr.content
                )

                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "content": text_result if text_result else "(see attached image)",
                    "tool_call_id": tr.tool_call_id,
                }
                if compat.get("requiresToolResultName") and tr.tool_name:
                    tool_msg["name"] = tr.tool_name
                params.append(tool_msg)

                if has_tool_images and "image" in model.input_types:
                    for c in tr.content:
                        if c.type == "image":
                            image_blocks.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{c.mime_type};base64,{c.data}",
                                },
                            })

                j += 1

            i = j - 1

            if image_blocks:
                if compat.get("requiresAssistantAfterToolResult"):
                    params.append({
                        "role": "assistant",
                        "content": "I have processed the tool results.",
                    })
                params.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Attached image(s) from tool result:",
                        },
                        *image_blocks,
                    ],
                })
                last_role = "user"
            else:
                last_role = "tool"

        i += 1

    return params


# =============================================================================
# 兼容性检测
# =============================================================================


def _detect_compat(model: Model) -> dict[str, Any]:
    """根据 provider 和 baseUrl 自动检测兼容性配置。

    Returns:
        完整解析的 compat dict。
    """
    provider = model.provider
    base_url = model.base_url

    is_zai = (
        provider in ("zai", "zai-coding-cn")
        or "api.z.ai" in base_url
        or "open.bigmodel.cn" in base_url
    )
    is_together = (
        provider == "together"
        or "api.together.ai" in base_url
        or "api.together.xyz" in base_url
    )
    is_moonshot = (
        provider in ("moonshotai", "moonshotai-cn")
        or "api.moonshot." in base_url
    )
    is_openrouter = provider == "openrouter" or "openrouter.ai" in base_url
    is_cf_workers_ai = (
        provider == "cloudflare-workers-ai"
        or "api.cloudflare.com" in base_url
    )
    is_cf_gateway = (
        provider == "cloudflare-ai-gateway"
        or "gateway.ai.cloudflare.com" in base_url
    )
    is_nvidia = (
        provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    )
    is_ant_ling = (
        provider == "ant-ling" or "api.ant-ling.com" in base_url
    )

    is_non_standard = (
        is_nvidia
        or provider == "cerebras"
        or "cerebras.ai" in base_url
        or provider == "xai"
        or "api.x.ai" in base_url
        or is_together
        or "chutes.ai" in base_url
        or "deepseek.com" in base_url
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cf_workers_ai
        or is_cf_gateway
        or is_ant_ling
    )

    use_max_tokens = (
        "chutes.ai" in base_url
        or is_moonshot
        or is_cf_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
    )

    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url
    is_openrouter_dev_role = is_openrouter and (
        model.id.startswith("anthropic/")
        or model.id.startswith("openai/")
    )

    cache_control_format = (
        "anthropic"
        if (provider == "openrouter" and model.id.startswith("anthropic/"))
        else None
    )

    # 确定 thinking_format
    if is_deepseek:
        thinking_format = "deepseek"
    elif is_zai:
        thinking_format = "zai"
    elif is_together:
        thinking_format = "together"
    elif is_ant_ling:
        thinking_format = "ant-ling"
    elif is_openrouter:
        thinking_format = "openrouter"
    else:
        thinking_format = "openai"

    return {
        "supportsStore": not is_non_standard,
        "supportsDeveloperRole": (
            is_openrouter_dev_role
            or (not is_non_standard and not is_openrouter)
        ),
        "supportsReasoningEffort": (
            not is_grok
            and not is_zai
            and not is_moonshot
            and not is_together
            and not is_cf_gateway
            and not is_nvidia
            and not is_ant_ling
        ),
        "supportsUsageInStreaming": True,
        "maxTokensField": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "requiresToolResultName": False,
        "requiresAssistantAfterToolResult": False,
        "requiresThinkingAsText": False,
        "requiresReasoningContentOnAssistantMessages": is_deepseek,
        "thinkingFormat": thinking_format,
        "zaiToolStream": False,
        "supportsStrictMode": (
            not is_moonshot
            and not is_together
            and not is_cf_gateway
            and not is_nvidia
        ),
        "cacheControlFormat": cache_control_format,
        "sendSessionAffinityHeaders": False,
        "supportsLongCacheRetention": not (
            is_together
            or is_cf_workers_ai
            or is_cf_gateway
            or is_nvidia
            or is_ant_ling
        ),
    }


def _get_compat(model: Model) -> dict[str, Any]:
    """获取模型的兼容性配置。

    model.compat 显式设置优先，否则 fallback 到 _detect_compat。
    """
    detected = _detect_compat(model)
    mc = model.compat
    if not mc:
        return detected

    return {
        k: mc.get(k, detected[k])
        for k in detected
    }


# =============================================================================
# 缓存控制
# =============================================================================


def _resolve_cache_retention(cache_retention: str | None = None) -> str:
    """解析缓存保留策略。

    优先级: 显式参数 > PI_CACHE_RETENTION 环境变量 > 默认 "short"
    """
    if cache_retention:
        return cache_retention
    if os.environ.get("PI_CACHE_RETENTION") == "long":
        return "long"
    return "short"


def _get_cache_control(
    compat: dict[str, Any], cache_retention: str
) -> dict[str, Any] | None:
    """根据 compat 和缓存策略计算 cache_control 对象。"""
    if compat.get("cacheControlFormat") != "anthropic" or cache_retention == "none":
        return None

    ttl = (
        "1h"
        if cache_retention == "long"
        and compat.get("supportsLongCacheRetention", True)
        else None
    )
    cc: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        cc["ttl"] = ttl
    return cc


def _apply_anthropic_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cache_control: dict[str, Any],
) -> None:
    """将 Anthropic-style cache_control 注入到请求中。"""
    _add_cache_control_to_system_prompt(messages, cache_control)
    _add_cache_control_to_last_tool(tools, cache_control)
    _add_cache_control_to_last_message(messages, cache_control)


def _add_cache_control_to_system_prompt(
    messages: list[dict[str, Any]],
    cache_control: dict[str, Any],
) -> None:
    for m in messages:
        if m.get("role") in ("system", "developer"):
            _add_cache_control_to_text_content(m, cache_control)
            return


def _add_cache_control_to_last_message(
    messages: list[dict[str, Any]],
    cache_control: dict[str, Any],
) -> None:
    for m in reversed(messages):
        if m.get("role") in ("user", "assistant"):
            if _add_cache_control_to_text_content(m, cache_control):
                return


def _add_cache_control_to_last_tool(
    tools: list[dict[str, Any]] | None,
    cache_control: dict[str, Any],
) -> None:
    if not tools or len(tools) == 0:
        return
    tools[-1]["cache_control"] = cache_control


def _add_cache_control_to_text_content(
    message: dict[str, Any],
    cache_control: dict[str, Any],
) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        if not content.strip():
            return False
        message["content"] = [{
            "type": "text",
            "text": content,
            "cache_control": cache_control,
        }]
        return True

    if not isinstance(content, list):
        return False

    for part in reversed(content):
        if isinstance(part, dict) and part.get("type") == "text":
            part["cache_control"] = cache_control
            return True

    return False


# =============================================================================
# 请求参数构建
# =============================================================================


def _build_params(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None,
    compat: dict[str, Any],
    cache_retention: str,
) -> dict[str, Any]:
    """构建 OpenAI Chat Completions 请求参数。"""
    messages = convert_messages(model, context, compat)
    cache_control = _get_cache_control(compat, cache_retention)

    params: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
    }

    # prompt_cache_key
    if (
        ("api.openai.com" in model.base_url and cache_retention != "none")
        or (
            cache_retention == "long"
            and compat.get("supportsLongCacheRetention", True)
        )
    ):
        session_id = options.session_id if options else None
        if session_id:
            params["prompt_cache_key"] = session_id[:64]

    # prompt_cache_retention
    if (
        cache_retention == "long"
        and compat.get("supportsLongCacheRetention", True)
    ):
        params["prompt_cache_retention"] = "24h"

    # stream_options
    if compat.get("supportsUsageInStreaming", True) is not False:
        params["stream_options"] = {"include_usage": True}

    # store
    if compat.get("supportsStore", True):
        params["store"] = False

    # max_tokens
    if options and options.max_tokens is not None:
        field = compat.get("maxTokensField", "max_completion_tokens")
        params[field] = options.max_tokens

    # temperature
    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    # tools
    if context.tools and len(context.tools) > 0:
        params["tools"] = _convert_tools(context.tools, compat)
        if compat.get("zaiToolStream"):
            params["tool_stream"] = True
    elif _has_tool_history(context.messages):
        # Anthropic via LiteLLM/proxy 要求有工具历史时必须传 tools
        params["tools"] = []

    # cache control
    if cache_control:
        _apply_anthropic_cache_control(
            messages, params.get("tools"), cache_control
        )

    # tool_choice
    if options and options.tool_choice:
        params["tool_choice"] = options.tool_choice

    # thinking/reasoning
    if model.reasoning:
        _apply_thinking_format(params, model, options, compat)

    return params


def _apply_thinking_format(
    params: dict[str, Any],
    model: Model,
    options: OpenAICompletionsOptions | None,
    compat: dict[str, Any],
) -> None:
    """按 compat.thinkingFormat 将 reasoning 参数注入到请求。

    支持: openai, deepseek, openrouter。其余格式静默跳过。
    """
    fmt = compat.get("thinkingFormat", "openai")
    has_effort = bool(options and options.reasoning_effort)
    off_value = (
        model.thinking_level_map.get("off")
        if model.thinking_level_map
        else None
    )

    if fmt == "openai":
        if has_effort and compat.get("supportsReasoningEffort", True):
            mapped = (
                model.thinking_level_map.get(options.reasoning_effort)
                if model.thinking_level_map
                else None
            )
            params["reasoning_effort"] = (
                mapped if isinstance(mapped, str) else options.reasoning_effort
            )
        elif (
            not has_effort
            and compat.get("supportsReasoningEffort", True)
            and isinstance(off_value, str)
        ):
            params["reasoning_effort"] = off_value

    elif fmt == "deepseek":
        params["thinking"] = {
            "type": "enabled" if has_effort else "disabled"
        }
        if has_effort and compat.get("supportsReasoningEffort", True):
            mapped = (
                model.thinking_level_map.get(options.reasoning_effort)
                if model.thinking_level_map
                else None
            )
            params["reasoning_effort"] = (
                mapped if isinstance(mapped, str) else options.reasoning_effort
            )

    elif fmt == "openrouter":
        if has_effort:
            mapped = (
                model.thinking_level_map.get(options.reasoning_effort)
                if model.thinking_level_map
                else None
            )
            params["reasoning"] = {
                "effort": (
                    mapped
                    if isinstance(mapped, str)
                    else options.reasoning_effort
                ),
            }
        elif off_value is not None:
            params["reasoning"] = {"effort": off_value}


# =============================================================================
# Tool 转换
# =============================================================================


def _convert_tools(
    tools: list[dict[str, Any]],
    compat: dict[str, Any],
) -> list[dict[str, Any]]:
    """将统一 Tool JSON Schema 列表转换为 OpenAI function 格式。"""
    result: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            schema = tool.get("parameters", {})
            converted: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": schema,
                },
            }
            if compat.get("supportsStrictMode", True) is not False:
                converted["function"]["strict"] = False
            result.append(converted)
    return result


# =============================================================================
# Stream chunk 处理
# =============================================================================


def _parse_chunk_usage(
    raw: dict[str, Any],
    model: Model,
) -> dict[str, Any]:
    """从 OpenAI chunk 中提取并规范化 token 用量。

    raw 格式: {"usage_obj": <ChatCompletionUsage object>}
    """
    usage_obj = raw.get("usage_obj")
    if usage_obj is None:
        return {}

    prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage_obj, "completion_tokens", 0) or 0

    # prompt_tokens_details
    details = getattr(usage_obj, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) if details else 0
    cache_write_tokens = (
        getattr(details, "cache_write_tokens", 0) if details else 0
    )

    # prompt_cache_hit_tokens fallback
    if not cached_tokens:
        cached_tokens = getattr(usage_obj, "prompt_cache_hit_tokens", 0) or 0

    cache_read = cached_tokens or 0
    cache_write = cache_write_tokens or 0
    input_tokens = max(0, prompt_tokens - cache_read - cache_write)

    cost = CostInfo()
    result = {
        "input": input_tokens,
        "output": completion_tokens,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "total_tokens": input_tokens + completion_tokens + cache_read + cache_write,
        "cost": cost,
    }

    # Build temp Usage for calculate_cost
    temp_usage = Usage(
        input=input_tokens,
        output=completion_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input_tokens + completion_tokens + cache_read + cache_write,
        cost=cost,
    )
    calculate_cost(model, temp_usage)
    return result


def _map_stop_reason(
    reason: str | None,
) -> tuple[str, str | None]:
    """将 OpenAI finish_reason 映射为 pimo-ai 标准 stop_reason。"""
    if reason is None:
        return "stop", None
    if reason in ("stop", "end"):
        return "stop", None
    if reason == "length":
        return "length", None
    if reason in ("function_call", "tool_calls"):
        return "toolUse", None
    if reason == "content_filter":
        return "error", "Provider finish_reason: content_filter"
    if reason == "network_error":
        return "error", "Provider finish_reason: network_error"
    return "error", f"Provider finish_reason: {reason}"


def _has_tool_history(messages: list[Message]) -> bool:
    """检查消息历史中是否包含工具调用或工具结果。"""
    for msg in messages:
        if msg.role == "toolResult":
            return True
        if msg.role == "assistant":
            for block in msg.content:
                if isinstance(block, ToolCall):
                    return True
    return False


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
    """构造 OpenAI SDK 客户端。

    根据 compat 自动配置 session affinity header。
    """
    headers: dict[str, str] = {}

    if session_id and compat.get("sendSessionAffinityHeaders"):
        headers["session_id"] = session_id
        headers["x-client-request-id"] = session_id
        headers["x-session-affinity"] = session_id

    if options_headers:
        headers.update(options_headers)

    return AsyncOpenAI(
        api_key=api_key,
        base_url=model.base_url,
        default_headers=headers if headers else None,
        max_retries=0,
    )
