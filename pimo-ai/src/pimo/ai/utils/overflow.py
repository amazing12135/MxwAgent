"""上下文溢出检测。

通过正则匹配各厂商的错误消息模式 + token 用量分析，判断一次 LLM 调用
是否因超出上下文窗口而失败。用于触发 auto-compaction。
"""

from __future__ import annotations

import re

from pimo.ai.types import AssistantMessage

# =============================================================================
# 溢出正则模式（23 个，覆盖 20+ 厂商）
# =============================================================================

_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"prompt is too long", re.IGNORECASE),            # Anthropic
    re.compile(r"request_too_large", re.IGNORECASE),             # Anthropic HTTP 413
    re.compile(r"input is too long for requested model", re.IGNORECASE),  # Bedrock
    re.compile(r"exceeds the context window", re.IGNORECASE),    # OpenAI
    re.compile(                                                  # LiteLLM
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
        re.IGNORECASE,
    ),
    re.compile(r"input token count.*exceeds the maximum", re.IGNORECASE),  # Gemini
    re.compile(r"maximum prompt length is \d+", re.IGNORECASE),  # xAI/Grok
    re.compile(r"reduce the length of the messages", re.IGNORECASE),  # Groq
    re.compile(r"maximum context length is \d+ tokens", re.IGNORECASE),  # OpenRouter
    re.compile(                                                  # OpenRouter/Poolside
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
        re.IGNORECASE,
    ),
    re.compile(                                                  # Together
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        re.IGNORECASE,
    ),
    re.compile(r"exceeds the limit of \d+", re.IGNORECASE),      # GitHub Copilot
    re.compile(r"exceeds the available context size", re.IGNORECASE),  # llama.cpp
    re.compile(r"greater than the context length", re.IGNORECASE),  # LM Studio
    re.compile(r"context window exceeds limit", re.IGNORECASE),  # MiniMax
    re.compile(r"exceeded model token limit", re.IGNORECASE),    # Kimi
    re.compile(r"too large for model with \d+ maximum context length", re.IGNORECASE),  # Mistral
    re.compile(r"model_context_window_exceeded", re.IGNORECASE),  # z.ai
    re.compile(r"prompt too long; exceeded (?:max )?context length", re.IGNORECASE),  # Ollama
    re.compile(r"context[_ ]length[_ ]exceeded", re.IGNORECASE),  # generic
    re.compile(r"too many tokens", re.IGNORECASE),                # generic
    re.compile(r"token limit exceeded", re.IGNORECASE),           # generic
    re.compile(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", re.IGNORECASE),  # Cerebras
]

# =============================================================================
# 排除模式（非溢出错误，3 个）
# =============================================================================

_NON_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(Throttling error|Service unavailable):", re.IGNORECASE),
    re.compile(r"rate limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
]


def is_context_overflow(
    message: AssistantMessage, context_window: int | None = None
) -> bool:
    """检查 AssistantMessage 是否表示上下文溢出错误。

    三种检测策略:
    1. 错误消息正则匹配（23 个模式，排除 throttle/rate-limit）
    2. 静默溢出: usage.input + cacheRead > contextWindow
    3. 长度截断: stopReason="length" + output=0 + input 填满窗口

    Args:
        message: LLM 返回的 AssistantMessage。
        context_window: 模型上下文窗口大小（token 数）。用于检测静默溢出。

    Returns:
        True 表示该消息由上下文溢出导致。
    """
    # Case 1: 错误消息匹配
    if message.stop_reason == "error" and message.error_message:
        err = message.error_message
        is_non_overflow = any(
            p.search(err) for p in _NON_OVERFLOW_PATTERNS
        )
        if not is_non_overflow:
            if any(p.search(err) for p in _OVERFLOW_PATTERNS):
                return True

    if context_window is None:
        return False

    input_tokens = message.usage.input + message.usage.cache_read

    # Case 2: 静默溢出（z.ai 风格）— 成功返回但用量超窗口
    if message.stop_reason == "stop" and input_tokens > context_window:
        return True

    # Case 3: 长度截断（MiMo 风格）— 输入填满窗口，output=0
    if (
        message.stop_reason == "length"
        and message.usage.output == 0
        and input_tokens >= context_window * 0.99
    ):
        return True

    return False


def get_overflow_patterns() -> list[re.Pattern[str]]:
    """获取溢出检测正则列表（用于测试）。

    Returns:
        正则模式列表的副本。
    """
    return list(_OVERFLOW_PATTERNS)
