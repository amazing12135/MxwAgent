"""pimo-ai 核心类型定义。

包含消息类型、内容块、模型描述、流式事件协议和调用参数。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

# =============================================================================
# 类型别名
# =============================================================================

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
"""思考/推理级别。不同厂商命名不同，pimo-ai 统一使用此枚举。"""

# =============================================================================
# 内容块 (Content Blocks)
# =============================================================================


@dataclass(kw_only=True)
class TextContent:
    """文本内容块。"""

    type: Literal["text"] = "text"
    text: str


@dataclass(kw_only=True)
class ThinkingContent:
    """思考过程内容块（对应 Claude extended thinking 等推理功能）。"""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None
    redacted: bool = False
    """True 表示思考内容被省略，仅保留加密签名。"""


@dataclass(kw_only=True)
class ImageContent:
    """图片内容块。data 为 base64 编码的图片数据。"""

    type: Literal["image"] = "image"
    data: str
    mime_type: str


@dataclass(kw_only=True)
class ToolCall:
    """LLM 请求的工具调用块。"""

    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any]

    @property
    def tool_call_id(self) -> str:
        """alias for id, 与 ToolResultMessage.tool_call_id 对应。"""
        return self.id


# =============================================================================
# Cost / Usage
# =============================================================================


@dataclass(kw_only=True)
class CostInfo:
    """token 成本信息（单位：美元）。"""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass(kw_only=True)
class Usage:
    """单次 LLM 调用的 token 用量和成本。"""

    input: int
    output: int
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: CostInfo = field(default_factory=CostInfo)


# =============================================================================
# 消息类型 (Messages)
# =============================================================================


@dataclass(kw_only=True)
class UserMessage:
    """用户消息。"""

    role: Literal["user"] = "user"
    content: list[TextContent | ImageContent]
    timestamp: float


@dataclass(kw_only=True)
class AssistantMessage:
    """助手（LLM）回复消息。

    包含完整的模型回复：文本、思考过程、工具调用。流式阶段为增量构建的
    partial 状态，最终 DoneEvent 携带完整定型消息。
    """

    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ThinkingContent | ImageContent | ToolCall]
    api: str
    provider: str
    model: str
    usage: Usage
    stop_reason: str
    error_message: str | None = None
    response_id: str | None = None
    """API 响应 ID (如 Anthropic message.id)，用于追踪."""
    timestamp: float = 0.0


@dataclass(kw_only=True)
class ToolResultMessage:
    """工具执行结果消息。"""

    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent]
    details: Any = None
    is_error: bool = False
    timestamp: float = 0.0


# 消息联合类型
Message = UserMessage | AssistantMessage | ToolResultMessage
"""LLM 可理解的标准消息。= UserMessage | AssistantMessage | ToolResultMessage"""

# =============================================================================
# 模型与上下文
# =============================================================================


@dataclass(kw_only=True)
class Model:
    """模型描述。包含协议、厂商、能力、计费信息。"""

    id: str
    """模型 ID，如 "claude-sonnet-4-20250514"."""
    name: str
    """人类可读名称，如 "Claude Sonnet 4"."""
    api: str
    """协议名，如 "anthropic-messages" / "openai-completions"."""
    provider: str
    """厂商名，如 "anthropic" / "openai" / "deepseek"."""
    base_url: str
    reasoning: bool
    """是否支持思考/推理功能."""
    input_types: list[str]
    """支持的输入类型，如 ["text", "image"]."""
    cost: CostInfo
    context_window: int
    """上下文窗口大小（token 数）."""
    max_tokens: int
    """单次最大输出 token 数."""
    thinking_level_map: dict[str, Any] | None = None
    """思考级别 → 厂商原生参数映射。None 表示不支持该级别."""
    compat: dict[str, Any] | None = None
    """厂商兼容性差异配置."""


@dataclass(kw_only=True)
class Tool:
    """LLM 可调用工具的基础定义。包含 JSON Schema 参数描述。

    AgentTool（定义在 pimo-agent-core）扩展此类，增加执行逻辑和 UI 元数据。
    """

    name: str
    """工具名称，LLM 通过此名称发起 toolCall。"""
    description: str
    """工具描述，帮助 LLM 决定何时调用。"""
    parameters: dict[str, Any]
    """JSON Schema 格式的参数定义。"""


@dataclass(kw_only=True)
class Context:
    """统一 LLM 调用上下文。"""

    system_prompt: str | None
    messages: list[Message]
    tools: list[Tool] | None = None
    """可用工具列表。为 None 表示不传递工具。"""

# =============================================================================
# 流式事件协议 (6 种统一事件)
# =============================================================================


@dataclass(kw_only=True)
class StartEvent:
    """流开始事件。携带初始 partial AssistantMessage（content 可能为空列表）。"""

    type: Literal["start"] = "start"
    message: AssistantMessage


@dataclass(kw_only=True)
class TextDeltaEvent:
    """文本增量事件。message 字段携带累积至今的 AssistantMessage 状态。"""

    type: Literal["text_delta"] = "text_delta"
    message: AssistantMessage


@dataclass(kw_only=True)
class ThinkingDeltaEvent:
    """思考增量事件。message 字段携带累积至今的 AssistantMessage 状态。"""

    type: Literal["thinking_delta"] = "thinking_delta"
    message: AssistantMessage


@dataclass(kw_only=True)
class ToolCallDeltaEvent:
    """工具调用增量事件。message 字段携带累积至今的 AssistantMessage 状态。"""

    type: Literal["toolcall_delta"] = "toolcall_delta"
    message: AssistantMessage


@dataclass(kw_only=True)
class DoneEvent:
    """流正常结束事件。携带完整、定型的 AssistantMessage。"""

    type: Literal["done"] = "done"
    message: AssistantMessage


@dataclass(kw_only=True)
class ErrorEvent:
    """流出错事件。message 为携带 stop_reason="error" 和 error_message 的 AssistantMessage。"""

    type: Literal["error"] = "error"
    message: AssistantMessage


# 事件联合类型
AssistantMessageEvent = (
    StartEvent | TextDeltaEvent | ThinkingDeltaEvent | ToolCallDeltaEvent | DoneEvent | ErrorEvent
)
"""Provider 流式响应归一化后的统一事件类型。"""

# =============================================================================
# 调用参数
# =============================================================================


@dataclass(kw_only=True)
class StreamOptions:
    """流式调用基础选项。"""

    api_key: str | None = None
    """API key。为 None 时从环境变量解析."""
    signal: Any = None
    """取消信号（如 asyncio.Event）。Provider 应定期检查."""
    transport: str = "auto"
    """传输方式：'auto' | 'sse' | 'websocket'."""
    temperature: float | None = None
    """采样温度。与 Anthropic extended thinking 互斥."""
    max_tokens: int | None = None
    """最大输出 token 数。为 None 时使用 model.max_tokens."""
    cache_retention: str | None = None
    """缓存保留策略：'none' | 'short' | 'long'."""
    session_id: str | None = None
    """会话 ID，用于 session-based caching."""
    headers: dict[str, str] | None = None
    """附加 HTTP 头。合并到 Provider 默认头之上."""
    timeout_ms: int | None = None
    """HTTP 请求超时（毫秒）."""
    websocket_connect_timeout_ms: int | None = None
    """WebSocket 连接超时（毫秒）."""
    max_retries: int | None = None
    """最大重试次数."""
    max_retry_delay_ms: int | None = None
    """Provider 请求重试延迟上限（毫秒）."""
    on_payload: Callable[[dict[str, Any]], None] | None = None
    """请求 payload 回调（用于日志/调试）."""
    on_response: Callable[[dict[str, Any]], None] | None = None
    """响应回调（用于日志/调试）."""
    metadata: dict[str, Any] | None = None
    """附加元数据。各厂商提取其理解的字段（如 user_id）."""


@dataclass(kw_only=True)
class SimpleStreamOptions(StreamOptions):
    """simplify 系列 API 的额外选项。封装 reasoning 差异，自动翻译为厂商原生参数。"""

    reasoning: ThinkingLevel | None = None
    """思考级别。None 表示不传 thinking 参数."""
    session_id: str | None = None
    """会话 ID，用于 LLM cache 亲和."""
    thinking_budgets: dict[str, int] | None = None
    """per-level thinking token 预算."""
