"""Unicode 清理工具（占位）。

TODO: 完整实现将在后续 /gen 任务中完成。
"""

from __future__ import annotations


def sanitize_surrogates(text: str) -> str:
    """移除 Unicode 代理项字符（surrogate characters）。

    防止 json.dumps 崩溃。

    Args:
        text: 输入文本。

    Returns:
        清理后的文本。
    """
    ...
