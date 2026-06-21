"""短哈希工具。

确定性快速的短哈希算法，将长字符串映射为短哈希值。
用于跨 provider 工具调用 ID 规范化。
"""

from __future__ import annotations


def short_hash(text: str) -> str:
    """生成短哈希值。

    使用 DJB 变种哈希算法，返回 base36 编码的 32-bit 哈希。
    相同输入始终产生相同输出。

    Args:
        text: 输入文本。

    Returns:
        base36 编码的短哈希字符串。
    """
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57

    for ch in text:
        code = ord(ch)
        h1 = ((h1 ^ code) * 2654435761) & 0xFFFFFFFF
        h2 = ((h2 ^ code) * 1597334677) & 0xFFFFFFFF

    h1 = (
        ((h1 ^ (h1 >> 16)) * 2246822507) & 0xFFFFFFFF
    ) ^ (
        ((h2 ^ (h2 >> 13)) * 3266489909) & 0xFFFFFFFF
    )
    h2 = (
        ((h2 ^ (h2 >> 16)) * 2246822507) & 0xFFFFFFFF
    ) ^ (
        ((h1 ^ (h1 >> 13)) * 3266489909) & 0xFFFFFFFF
    )

    # 转无符号 32-bit → base36
    def _to_base36(n: int) -> str:
        n = n & 0xFFFFFFFF
        if n == 0:
            return "0"
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        result = ""
        while n > 0:
            result = chars[n % 36] + result
            n //= 36
        return result

    return _to_base36(h2) + _to_base36(h1)
