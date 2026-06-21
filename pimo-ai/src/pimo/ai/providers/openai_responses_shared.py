"""OpenAI Responses API 共享模块。

提供 convert_responses_messages、convert_responses_tools、process_responses_stream
等函数，被 openai_responses.py 和 openai_codex_responses.py 复用。
"""

from __future__ import annotations

import json as _json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pimo.ai.event_stream import AssistantMessageEventStream
from pimo.ai.models.cost import calculate_cost
from pimo.ai.providers.transform_messages import transform_messages
from pimo.ai.types import (
    AssistantMessage,
    Context,
    CostInfo,
    ImageContent,
    Message,
    Model,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolCallDeltaEvent,
    Usage,
)
from pimo.ai.utils.hash import short_hash
from pimo.ai.utils.json_parse import parse_streaming_json
from pimo.ai.utils.sanitize_unicode import sanitize_surrogates

# =============================================================================
# 类型定义
# =============================================================================


@dataclass
class OpenAIResponsesStreamOptions:
    """process_responses_stream 的选项。"""

    service_tier: str | None = None
    """请求时指定的 service tier。"""

    resolve_service_tier: Callable[
        [str | None, str | None], str | None
    ] | None = None
    """解析最终 service tier 的回调。"""

    apply_service_tier_pricing: Callable[
        [Usage, str | None], None
    ] | None = None
    """应用 service tier 定价倍率的回调。"""


@dataclass
class ConvertResponsesMessagesOptions:
    """convert_responses_messages 的选项。"""

    include_system_prompt: bool = True
    """是否在转换结果中包含 system prompt。"""


@dataclass
class ConvertResponsesToolsOptions:
    """convert_responses_tools 的选项。"""

    strict: bool | None = None
    """是否启用 strict 模式。None 时默认 False。"""


# =============================================================================
# 文本签名编码（跨轮文本块 ID 追踪）
# =============================================================================


def _encode_text_signature_v1(
    id: str, phase: str | None = None
) -> str:
    """编码 V1 版本文本块签名。

    Args:
        id: 消息 ID。
        phase: 可选的消息阶段（"commentary" / "final_answer"）。

    Returns:
        JSON 字符串格式的签名。
    """
    payload: dict[str, Any] = {"v": 1, "id": id}
    if phase:
        payload["phase"] = phase
    return _json.dumps(payload)


def _parse_text_signature(
    signature: str | None,
) -> dict[str, str] | None:
    """解析文本块签名。

    支持 V1 JSON 格式和 legacy 纯字符串格式。

    Args:
        signature: 签名文本。

    Returns:
        {"id": str, "phase"?: str} 或 None。
    """
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            parsed = _json.loads(signature)
            if (
                isinstance(parsed, dict)
                and parsed.get("v") == 1
                and isinstance(parsed.get("id"), str)
            ):
                result: dict[str, str] = {"id": parsed["id"]}
                phase = parsed.get("phase")
                if phase in ("commentary", "final_answer"):
                    result["phase"] = phase
                return result
        except (_json.JSONDecodeError, TypeError):
            pass
    # legacy: plain string id
    return {"id": signature}


# =============================================================================
# Message 转换
# =============================================================================


def convert_responses_messages(
    model: Model,
    context: Context,
    allowed_tool_call_providers: set[str] | None = None,
    *,
    include_system_prompt: bool = True,
) -> list[dict[str, Any]]:
    """将统一 Message 列表转换为 OpenAI Responses Input 格式。

    Args:
        model: 目标模型。
        context: 统一上下文。
        allowed_tool_call_providers: 允许 pipe 格式 tool call ID 直传的 provider 集合。
        include_system_prompt: 是否包含 system prompt。

    Returns:
        OpenAI ResponseInput 列表。
    """
    providers = allowed_tool_call_providers or set()
    messages: list[dict[str, Any]] = []

    # ---- 内部辅助 ----

    def _normalize_id_part(part: str) -> str:
        sanitized = "".join(
            c if (c.isascii() and (c.isalnum() or c in "_-")) else "_"
            for c in part
        )
        normalized = sanitized[:64] if len(sanitized) > 64 else sanitized
        return normalized.rstrip("_")

    def _build_foreign_responses_item_id(item_id: str) -> str:
        normalized = f"fc_{short_hash(item_id)}"
        return normalized[:64] if len(normalized) > 64 else normalized

    def _normalize_tool_call_id(
        id: str, _target_model: Model, source: Message,
    ) -> str:
        if model.provider not in providers:
            return _normalize_id_part(id)
        if "|" not in id:
            return _normalize_id_part(id)
        call_id, item_id = id.split("|", 1)
        normalized_call_id = _normalize_id_part(call_id)
        is_foreign = (
            getattr(source, "provider", None) != model.provider
            or getattr(source, "api", None) != model.api
        )
        normalized_item_id = (
            _build_foreign_responses_item_id(item_id)
            if is_foreign
            else _normalize_id_part(item_id)
        )
        if not normalized_item_id.startswith("fc_"):
            normalized_item_id = _normalize_id_part(
                f"fc_{normalized_item_id}"
            )
        return f"{normalized_call_id}|{normalized_item_id}"

    # 跨 provider 消息规范化
    transformed_messages = transform_messages(
        context.messages, model, _normalize_tool_call_id,
    )

    # System prompt
    if include_system_prompt and context.system_prompt:
        compat = model.compat or {}
        supports_dev = compat.get("supportsDeveloperRole", True)
        role = (
            "developer"
            if (model.reasoning and supports_dev is not False)
            else "system"
        )
        messages.append({
            "role": role,
            "content": sanitize_surrogates(context.system_prompt),
        })

    # 逐消息转换
    msg_index = 0
    for msg in transformed_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": sanitize_surrogates(msg.content),
                        }
                    ],
                })
            elif isinstance(msg.content, list):
                content_parts: list[dict[str, Any]] = []
                for item in msg.content:
                    if getattr(item, "type", None) == "text":
                        content_parts.append({
                            "type": "input_text",
                            "text": sanitize_surrogates(item.text),
                        })
                    elif getattr(item, "type", None) == "image":
                        content_parts.append({
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": (
                                f"data:{item.mime_type};base64,{item.data}"
                            ),
                        })
                if content_parts:
                    messages.append({
                        "role": "user",
                        "content": content_parts,
                    })

        elif msg.role == "assistant":
            output_items: list[dict[str, Any]] = []
            assistant_msg = msg
            is_different_model = (
                getattr(assistant_msg, "model", None) != model.id
                and getattr(assistant_msg, "provider", None) == model.provider
                and getattr(assistant_msg, "api", None) == model.api
            )
            text_block_index = 0

            for block in msg.content:
                if isinstance(block, ThinkingContent):
                    # 通过 thinkingSignature 重放 reasoning item
                    if block.signature:
                        try:
                            reasoning_item = _json.loads(block.signature)
                            output_items.append(reasoning_item)
                        except (_json.JSONDecodeError, TypeError):
                            pass

                elif isinstance(block, TextContent):
                    parsed_sig = _parse_text_signature(
                        getattr(block, "textSignature", None)
                    )
                    fallback_id = (
                        f"msg_pi_{msg_index}"
                        if text_block_index == 0
                        else f"msg_pi_{msg_index}_{text_block_index}"
                    )
                    text_block_index += 1

                    msg_id = parsed_sig["id"] if parsed_sig else None
                    if not msg_id:
                        msg_id = fallback_id
                    elif len(msg_id) > 64:
                        msg_id = f"msg_{short_hash(msg_id)}"

                    phase = (
                        parsed_sig.get("phase") if parsed_sig else None
                    )

                    output_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": sanitize_surrogates(block.text),
                                "annotations": [],
                            }
                        ],
                        "status": "completed",
                        "id": msg_id,
                        **({"phase": phase} if phase else {}),
                    })

                elif isinstance(block, ToolCall):
                    parts = block.id.split("|", 1)
                    call_id = parts[0]
                    item_id: str | None = (
                        parts[1] if len(parts) > 1 else None
                    )

                    # 跨模型消息：省略 item_id 避免 OpenAI pairing 校验
                    if (
                        is_different_model
                        and item_id
                        and item_id.startswith("fc_")
                    ):
                        item_id = None

                    fc_item: dict[str, Any] = {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": block.name,
                        "arguments": _json.dumps(block.arguments),
                    }
                    if item_id:
                        fc_item["id"] = item_id
                    output_items.append(fc_item)

            if output_items:
                messages.extend(output_items)

        elif msg.role == "toolResult":
            text_result = "\n".join(
                c.text
                for c in msg.content
                if isinstance(c, TextContent)
            )
            has_images = any(
                getattr(c, "type", None) == "image"
                for c in msg.content
            )
            has_text = len(text_result) > 0

            call_parts = msg.tool_call_id.split("|", 1)
            call_id = call_parts[0]

            if has_images and "image" in model.input_types:
                content_parts: list[dict[str, Any]] = []
                if has_text:
                    content_parts.append({
                        "type": "input_text",
                        "text": sanitize_surrogates(text_result),
                    })
                for block in msg.content:
                    if getattr(block, "type", None) == "image":
                        content_parts.append({
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": (
                                f"data:{block.mime_type};base64,{block.data}"
                            ),
                        })
                output_content: str | list[dict[str, Any]] = content_parts
            else:
                output_content = sanitize_surrogates(
                    text_result if has_text else "(see attached image)"
                )

            messages.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_content,
            })

        msg_index += 1

    return messages


# =============================================================================
# Tool 转换
# =============================================================================


def convert_responses_tools(
    tools: list[dict[str, Any]],
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """将统一 Tool JSON Schema 列表转换为 OpenAI Responses Tool 格式。

    Args:
        tools: JSON Schema 格式的工具定义列表。
        strict: 是否启用 strict 模式。

    Returns:
        OpenAI Tool 数组。
    """
    result: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        result.append({
            "type": "function",
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
            "strict": strict,
        })
    return result


# =============================================================================
# Stream 处理
# =============================================================================


async def process_responses_stream(
    openai_stream: Any,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    options: OpenAIResponsesStreamOptions | None = None,
) -> None:
    """消费 OpenAI Responses SSE 事件流，转换为归一化事件。

    处理的 13 种事件类型:
    - response.created / output_item.added / reasoning_summary_part.added
    - reasoning_summary_text.delta / reasoning_summary_part.done
    - reasoning_text.delta / content_part.added
    - output_text.delta / refusal.delta
    - function_call_arguments.delta / function_call_arguments.done
    - output_item.done / response.completed
    - error / response.failed

    Args:
        openai_stream: OpenAI SDK 返回的 ResponseStreamEvent 异步迭代器。
        output: 正在构建的 AssistantMessage（原地修改）。
        stream: AssistantMessageEventStream 生产者。
        model: 目标模型。
        options: 流选项。

    Raises:
        RuntimeError: 当收到 error 或 response.failed 事件时。
    """
    current_item: dict[str, Any] | None = None
    current_block: (
        ThinkingContent
        | TextContent
        | ToolCall
        | None
    ) = None
    _partial_json: str = ""

    def _push_text_delta() -> None:
        stream.push(TextDeltaEvent(message=output))

    def _push_thinking_delta() -> None:
        stream.push(ThinkingDeltaEvent(message=output))

    def _push_toolcall_delta() -> None:
        stream.push(ToolCallDeltaEvent(message=output))

    async for event in openai_stream:
        event_type = getattr(event, "type", None)

        # ---- response.created ----
        if event_type == "response.created":
            response = getattr(event, "response", None)
            if response:
                output.response_id = getattr(response, "id", None)

        # ---- response.output_item.added ----
        elif event_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if not item:
                continue
            item_type = getattr(item, "type", None)

            if item_type == "reasoning":
                current_item = item
                current_block = ThinkingContent(thinking="")
                output.content.append(current_block)

            elif item_type == "message":
                current_item = item
                current_block = TextContent(text="")
                output.content.append(current_block)

            elif item_type == "function_call":
                current_item = item
                call_id = getattr(item, "call_id", "")
                item_id = getattr(item, "id", "")
                name = getattr(item, "name", "")
                initial_args = getattr(item, "arguments", "") or ""
                _partial_json = initial_args

                current_block = ToolCall(
                    id=f"{call_id}|{item_id}",
                    name=name,
                    arguments={},
                )
                output.content.append(current_block)

        # ---- response.reasoning_summary_part.added ----
        elif event_type == "response.reasoning_summary_part.added":
            if current_item and current_item.get("type") == "reasoning":
                part = getattr(event, "part", None)
                if part:
                    summary = current_item.setdefault("summary", [])
                    summary.append(part)

        # ---- response.reasoning_summary_text.delta ----
        elif event_type == "response.reasoning_summary_text.delta":
            if (
                current_item
                and current_item.get("type") == "reasoning"
                and isinstance(current_block, ThinkingContent)
            ):
                summary = current_item.setdefault("summary", [])
                if summary:
                    last_part = summary[-1]
                    delta = getattr(event, "delta", "")
                    current_block.thinking += delta
                    last_part["text"] = last_part.get("text", "") + delta
                    _push_thinking_delta()

        # ---- response.reasoning_summary_part.done ----
        elif event_type == "response.reasoning_summary_part.done":
            if (
                current_item
                and current_item.get("type") == "reasoning"
                and isinstance(current_block, ThinkingContent)
            ):
                summary = current_item.setdefault("summary", [])
                if summary:
                    last_part = summary[-1]
                    current_block.thinking += "\n\n"
                    last_part["text"] = last_part.get("text", "") + "\n\n"
                    _push_thinking_delta()

        # ---- response.reasoning_text.delta ----
        elif event_type == "response.reasoning_text.delta":
            if (
                current_item
                and current_item.get("type") == "reasoning"
                and isinstance(current_block, ThinkingContent)
            ):
                delta = getattr(event, "delta", "")
                current_block.thinking += delta
                _push_thinking_delta()

        # ---- response.content_part.added ----
        elif event_type == "response.content_part.added":
            if current_item and current_item.get("type") == "message":
                part = getattr(event, "part", None)
                if part:
                    part_type = getattr(part, "type", None)
                    if part_type in ("output_text", "refusal"):
                        content_list = current_item.setdefault(
                            "content", []
                        )
                        content_list.append(part)

        # ---- response.output_text.delta ----
        elif event_type == "response.output_text.delta":
            if (
                current_item
                and current_item.get("type") == "message"
                and isinstance(current_block, TextContent)
            ):
                content_list = current_item.get("content", [])
                if not content_list:
                    continue
                last_part = content_list[-1]
                if last_part.get("type") == "output_text":
                    delta = getattr(event, "delta", "")
                    current_block.text += delta
                    last_part["text"] = last_part.get("text", "") + delta
                    _push_text_delta()

        # ---- response.refusal.delta ----
        elif event_type == "response.refusal.delta":
            if (
                current_item
                and current_item.get("type") == "message"
                and isinstance(current_block, TextContent)
            ):
                content_list = current_item.get("content", [])
                if not content_list:
                    continue
                last_part = content_list[-1]
                if last_part.get("type") == "refusal":
                    delta = getattr(event, "delta", "")
                    current_block.text += delta
                    last_part["refusal"] = (
                        last_part.get("refusal", "") + delta
                    )
                    _push_text_delta()

        # ---- response.function_call_arguments.delta ----
        elif event_type == "response.function_call_arguments.delta":
            if (
                current_item
                and current_item.get("type") == "function_call"
                and isinstance(current_block, ToolCall)
            ):
                delta = getattr(event, "delta", "")
                _partial_json += delta
                current_block.arguments = parse_streaming_json(
                    _partial_json
                )
                _push_toolcall_delta()

        # ---- response.function_call_arguments.done ----
        elif event_type == "response.function_call_arguments.done":
            if (
                current_item
                and current_item.get("type") == "function_call"
                and isinstance(current_block, ToolCall)
            ):
                previous = _partial_json
                full_args = getattr(event, "arguments", "")
                _partial_json = full_args
                current_block.arguments = parse_streaming_json(
                    _partial_json
                )

                # 推送尾差（final delta）
                if full_args.startswith(previous):
                    tail_delta = full_args[len(previous):]
                    if tail_delta:
                        _push_toolcall_delta()

        # ---- response.output_item.done ----
        elif event_type == "response.output_item.done":
            item = getattr(event, "item", None)
            if not item:
                continue
            item_type = getattr(item, "type", None)

            if (
                item_type == "reasoning"
                and isinstance(current_block, ThinkingContent)
            ):
                summary_list = item.get("summary") or []
                content_list = item.get("content") or []
                summary_text = "\n\n".join(
                    s.get("text", "") for s in summary_list
                )
                content_text = "\n\n".join(
                    c.get("text", "") for c in content_list
                )
                current_block.thinking = (
                    summary_text or content_text or current_block.thinking
                )
                current_block.signature = _json.dumps(item)
                current_block = None

            elif (
                item_type == "message"
                and isinstance(current_block, TextContent)
            ):
                parts = item.get("content") or []
                current_block.text = "".join(
                    p.get("text", "") if p.get("type") == "output_text"
                    else p.get("refusal", "")
                    for p in parts
                )
                # Encode text signature for cross-turn tracking
                phase = item.get("phase")
                current_block.__dict__["textSignature"] = (
                    _encode_text_signature_v1(
                        item.get("id", ""),
                        phase if phase else None,
                    )
                )
                current_block = None

            elif (
                item_type == "function_call"
                and isinstance(current_block, ToolCall)
            ):
                args_raw = (
                    _partial_json
                    if _partial_json
                    else (item.get("arguments") or "{}")
                )
                current_block.arguments = parse_streaming_json(args_raw)
                current_block = None
                _partial_json = ""

        # ---- response.completed ----
        elif event_type == "response.completed":
            response = getattr(event, "response", None)
            if response:
                rid = getattr(response, "id", None)
                if rid:
                    output.response_id = rid
                resp_usage = getattr(response, "usage", None)
                if resp_usage:
                    input_tokens = getattr(resp_usage, "input_tokens", 0) or 0
                    output_tokens = getattr(resp_usage, "output_tokens", 0) or 0
                    total_tokens = getattr(resp_usage, "total_tokens", 0) or 0

                    details = getattr(
                        resp_usage, "input_tokens_details", None
                    )
                    cached_tokens = (
                        getattr(details, "cached_tokens", 0) if details else 0
                    )

                    output.usage = Usage(
                        input=(input_tokens - cached_tokens),
                        output=output_tokens,
                        cache_read=cached_tokens or 0,
                        cache_write=0,
                        total_tokens=total_tokens,
                        cost=CostInfo(),
                    )

                    calculate_cost(model, output.usage)

                    # service tier pricing
                    if options and options.apply_service_tier_pricing:
                        response_tier = getattr(
                            response, "service_tier", None
                        )
                        tier = (
                            options.resolve_service_tier(
                                response_tier, options.service_tier,
                            )
                            if options.resolve_service_tier
                            else (response_tier or options.service_tier)
                        )
                        options.apply_service_tier_pricing(
                            output.usage, tier
                        )

                # stop reason
                output.stop_reason = _map_stop_reason(
                    getattr(response, "status", None)
                )
                if (
                    any(isinstance(b, ToolCall) for b in output.content)
                    and output.stop_reason == "stop"
                ):
                    output.stop_reason = "toolUse"

        # ---- error ----
        elif event_type == "error":
            code = getattr(event, "code", "unknown")
            msg = getattr(event, "message", "Unknown error")
            raise RuntimeError(f"Error Code {code}: {msg}")

        # ---- response.failed ----
        elif event_type == "response.failed":
            response = getattr(event, "response", None)
            err = getattr(response, "error", None) if response else None
            details = (
                getattr(response, "incomplete_details", None)
                if response
                else None
            )
            if err:
                code = getattr(err, "code", "unknown")
                message = getattr(err, "message", "no message")
                raise RuntimeError(f"{code}: {message}")
            elif details:
                reason = getattr(details, "reason", "unknown")
                raise RuntimeError(f"incomplete: {reason}")
            else:
                raise RuntimeError(
                    "Unknown error (no error details in response)"
                )


# =============================================================================
# Stop Reason 映射
# =============================================================================


def _map_stop_reason(status: str | None) -> str:
    """将 OpenAI Responses response.status 映射为 pimo-ai 统一 stop_reason。

    映射:
        completed / in_progress / queued → "stop"
        incomplete → "length"
        failed / cancelled → "error"

    Args:
        status: OpenAI response.status 值。

    Returns:
        标准化 stop_reason。
    """
    if not status:
        return "stop"
    if status == "completed":
        return "stop"
    if status == "incomplete":
        return "length"
    if status in ("failed", "cancelled"):
        return "error"
    if status in ("in_progress", "queued"):
        return "stop"
    raise ValueError(f"Unhandled stop reason: {status}")
