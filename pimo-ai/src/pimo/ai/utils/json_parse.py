"""JSON 解析修复工具。

LLM 流式输出 JSON 时常见问题:
- 流式截断（不完整的 JSON）
- 字符串内的未转义控制字符（\\n、\\t 等原始字符而非转义序列）
- 无效的转义序列（如 \\s）
- 结尾多余的逗号

这些函数用多层回退策略修复上述问题。
"""

from __future__ import annotations

import json
import re
from typing import Any

# 合法的 JSON 转义字符
_VALID_JSON_ESCAPES = frozenset({'"', "\\", "/", "b", "f", "n", "r", "t", "u"})


def _is_control_char(ch: str) -> bool:
    """检查字符是否为 ASCII 控制字符 (0x00-0x1F)。"""
    cp = ord(ch)
    return 0x00 <= cp <= 0x1F


def _escape_control_char(ch: str) -> str:
    """将控制字符转义为 JSON 转义序列。"""
    cp = ord(ch)
    mapping = {
        0x08: "\\b",
        0x0C: "\\f",
        0x0A: "\\n",
        0x0D: "\\r",
        0x09: "\\t",
    }
    if cp in mapping:
        return mapping[cp]
    return f"\\u{cp:04x}"


def _repair_json(json_str: str) -> str:
    """修复畸形 JSON 字符串中的常见问题。

    修复:
    - 字符串内的原始控制字符 → 转义序列
    - 无效转义序列前的反斜杠 → 加倍（"\\\\")

    Args:
        json_str: 可能是畸形的 JSON 文本。

    Returns:
        修复后的 JSON 文本。
    """
    result: list[str] = []
    in_string = False
    i = 0

    while i < len(json_str):
        ch = json_str[i]

        if not in_string:
            result.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        if ch == '"':
            result.append(ch)
            in_string = False
            i += 1
            continue

        if ch == "\\":
            next_ch = json_str[i + 1] if i + 1 < len(json_str) else None

            if next_ch is None:
                # 末尾孤立反斜杠 → 加倍
                result.append("\\\\")
                i += 1
                continue

            if next_ch == "u" and i + 5 < len(json_str):
                hex_digits = json_str[i + 2:i + 6]
                if re.fullmatch(r"[0-9a-fA-F]{4}", hex_digits):
                    result.append(f"\\u{hex_digits}")
                    i += 6
                    continue

            if next_ch in _VALID_JSON_ESCAPES:
                result.append(f"\\{next_ch}")
                i += 2
                continue

            # 无效转义 → 加倍反斜杠
            result.append("\\\\")
            i += 1
            continue

        # 字符串内的原始控制字符 → 转义
        if _is_control_char(ch):
            result.append(_escape_control_char(ch))
        else:
            result.append(ch)
        i += 1

    return "".join(result)


def parse_json_with_repair(json_str: str) -> Any:
    """解析 JSON 并尝试修复常见错误。

    先直接解析，失败则 repair 后重试。

    Args:
        json_str: JSON 文本。

    Returns:
        解析后的 Python 对象。

    Raises:
        json.JSONDecodeError: 修复后仍无法解析。
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        repaired = _repair_json(json_str)
        if repaired != json_str:
            return json.loads(repaired)
        raise


def _try_complete_json(text: str) -> Any | None:
    """尝试通过补齐括号/引号来修复不完整的 JSON。

    Args:
        text: 可能是截断的 JSON 文本。

    Returns:
        解析后的对象，或 None（无法修复）。
    """
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None

    # 移除末尾逗号（常见于流式输出中间截断）
    while stripped.endswith(",") or stripped.endswith(":"):
        stripped = stripped[:-1].strip()

    # 计算未闭合的括号
    stack: list[str] = []
    in_string = False
    i = 0
    while i < len(stripped):
        ch = stripped[i]
        if ch == "\\" and in_string:
            i += 2
            continue
        if ch == '"' and not in_string:
            in_string = True
        elif ch == '"' and in_string:
            in_string = False
        elif not in_string:
            if ch in "{[":
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()
        i += 1

    # 补齐未闭合的括号/引号
    completed = stripped
    if in_string:
        completed += '"'
    for opener in reversed(stack):
        completed += "}" if opener == "{" else "]"

    try:
        return json.loads(completed)
    except json.JSONDecodeError:
        return None


def parse_streaming_json(text: str | None) -> dict[str, Any]:
    """解析流式（可能不完整）的 JSON 文本。

    多层回退:
    1. parse_json_with_repair
    2. 括号补齐
    3. repair + 括号补齐
    4. 返回 {}

    永不抛异常——流式场景下静默返回空 dict。

    Args:
        text: 待解析的 JSON 文本。

    Returns:
        解析后的 dict。解析失败返回空 dict。
    """
    if not text or not text.strip():
        return {}

    # Layer 1: direct parse + repair
    try:
        return parse_json_with_repair(text)
    except json.JSONDecodeError:
        pass

    # Layer 2: complete JSON by closing braces
    completed = _try_complete_json(text)
    if completed is not None:
        return completed if isinstance(completed, dict) else {}

    # Layer 3: repair + complete
    try:
        repaired = _repair_json(text)
        if repaired != text:
            result = _try_complete_json(repaired)
            if result is not None:
                return result if isinstance(result, dict) else {}
    except Exception:
        pass

    # Layer 4: give up
    return {}
