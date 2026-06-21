"""跨 Provider 共享的 SimpleStreamOptions 辅助函数。

提供 SimpleStreamOptions → StreamOptions 字段展开，以及 thinking 预算
与 max_tokens 的联合计算。"""

from __future__ import annotations

from pimo.ai.types import (
    Model,
    SimpleStreamOptions,
    StreamOptions,
    ThinkingLevel,
)

# 默认 thinking 预算（按级别）
_DEFAULT_BUDGETS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}

_MIN_OUTPUT_TOKENS = 1024


def build_base_options(
    model: Model,
    options: SimpleStreamOptions | None = None,
    api_key: str | None = None,
) -> StreamOptions:
    """将 SimpleStreamOptions 展开为 StreamOptions。

    提取 SimpleStreamOptions 中与 StreamOptions 共有字段的值，
    api_key 优先使用显式参数，fallback 到 options.api_key。

    Args:
        model: 目标模型（当前未使用，保留签名对齐）。
        options: 简化选项。
        api_key: API Key。优先级高于 options.api_key。

    Returns:
        字段已填充的 StreamOptions 实例。
    """
    opts = options
    return StreamOptions(
        temperature=opts.temperature if opts else None,
        max_tokens=opts.max_tokens if opts else None,
        signal=opts.signal if opts else None,
        api_key=api_key or (opts.api_key if opts else None),
        transport=opts.transport if opts else "auto",
        cache_retention=opts.cache_retention if opts else None,
        session_id=opts.session_id if opts else None,
        headers=opts.headers if opts else None,
        timeout_ms=opts.timeout_ms if opts else None,
        websocket_connect_timeout_ms=(
            opts.websocket_connect_timeout_ms if opts else None
        ),
        max_retries=opts.max_retries if opts else None,
        max_retry_delay_ms=opts.max_retry_delay_ms if opts else None,
        on_payload=opts.on_payload if opts else None,
        on_response=opts.on_response if opts else None,
        metadata=opts.metadata if opts else None,
    )


def _clamp_reasoning(
    effort: ThinkingLevel | None,
) -> ThinkingLevel | None:
    """将 "xhigh" 夹紧为 "high"。

    非自适应思考模型不支持 "xhigh" 级别，需要降级。

    Args:
        effort: 原始思考级别。

    Returns:
        夹紧后的级别。xhigh→high，其他原样返回。
    """
    if effort == "xhigh":
        return "high"
    return effort


def adjust_max_tokens_for_thinking(
    base_max_tokens: int | None,
    model_max_tokens: int,
    reasoning_level: ThinkingLevel,
    custom_budgets: dict[str, int] | None = None,
) -> dict[str, int]:
    """为非自适应模型计算 max_tokens 和 thinking token 预算。

    将 thinking budget 嵌入 max_tokens 空间，当两者接近时保留最小输出容量。

    Args:
        base_max_tokens: 调用方显式设置的 max_tokens。None 表示使用模型上限。
        model_max_tokens: 模型最大输出 token 数。
        reasoning_level: pimo ThinkingLevel。
        custom_budgets: per-level thinking token 预算覆写。

    Returns:
        ``{"max_tokens": int, "thinking_budget": int}``
    """
    budgets = {**_DEFAULT_BUDGETS, **(custom_budgets or {})}

    # clamp xhigh → high
    level = _clamp_reasoning(reasoning_level) or "high"

    thinking_budget = budgets.get(level, _DEFAULT_BUDGETS["high"])

    if base_max_tokens is None:
        max_tokens = model_max_tokens
    else:
        max_tokens = min(base_max_tokens + thinking_budget, model_max_tokens)

    # 预算超出可用空间时压缩
    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - _MIN_OUTPUT_TOKENS)

    return {"max_tokens": max_tokens, "thinking_budget": thinking_budget}
