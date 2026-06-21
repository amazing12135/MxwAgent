"""Unicode 代理项清理。

移除字符串中的未配对 Unicode 代理项字符（unpaired surrogates），
防止 json.dumps 等序列化操作崩溃。合法的 emoji 等代理对不受影响。
"""

from __future__ import annotations

import re

# 匹配未配对的代理项:
# - 高代理项 (0xD800-0xDBFF) 后面不跟低代理项 (0xDC00-0xDFFF)
# - 低代理项 (0xDC00-0xDFFF) 前面没有高代理项
_SURROGATE_PATTERN = re.compile(
    r"[\uD800-\uDBFF](?![\uDC00-\uDFFF])"
    r"|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]"
)


def sanitize_surrogates(text: str) -> str:
    """移除未配对的 Unicode 代理项字符。

    合法的 emoji（如 🙈）使用正确的代理对，不会被移除。

    Args:
        text: 输入文本。

    Returns:
        清理后的文本。
    """
    return _SURROGATE_PATTERN.sub("", text)
