"""pimo-agent-core 核心类型定义。

包含 Agent 级别的消息联合类型、工具定义、状态、事件协议和 Agent Loop 配置。
Harness 层类型（FileSystem, SessionTreeEntry, AgentHarnessEvent 等）定义在
``pimo.agent_core.harness.types``。

自定义消息类型（BashExecution, CompactionSummary, BranchSummary, Custom）定义在
``pimo.agent_core.harness.messages``，此处通过字符串前向引用引用。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar, Union

if TYPE_CHECKING:
    from pimo.ai.event_stream import AssistantMessageEventStream

from pimo.ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    ThinkingLevel,
    Tool,
    ToolCall,
    ToolResultMessage,
)

# =============================================================================
# 取消信号
# =============================================================================

AbortSignal = asyncio.Event
"""取消信号类型别名。Agent Loop 中用于取消正在进行的 LLM 调用或工具执行。"""

# =============================================================================
# 基础类型别名
# =============================================================================

ToolExecutionMode: TypeAlias = Literal["sequential", "parallel"]
"""工具执行模式。

- ``"sequential"``: 每个 tool call 依次 prepare → execute → finalize。
- ``"parallel"``: 所有 tool call 先依次 prepare，再并发 execute，最后依序 finalize。
"""

QueueMode: TypeAlias = Literal["all", "one-at-a-time"]
"""队列消费模式，控制 steering/follow-up 队列中消息的注入方式。

- ``"all"``: 一次性排空并注入所有排队消息。
- ``"one-at-a-time"``: 每次只注入最早的一条排队消息。
"""

StreamFn: TypeAlias = Callable[
    [Model, "AgentContext", SimpleStreamOptions | None],
    Awaitable["AssistantMessageEventStream"],
]
"""Agent Loop 使用的流函数签名。

约定:
- 不得抛出异常或返回 rejected promise（请求/模型/运行时失败均编码在流中）。
- 必须返回 AssistantMessageEventStream。
- 失败通过 stopReason="error" 或 "aborted" 的最终 AssistantMessage 体现。
"""

AgentToolCall: TypeAlias = ToolCall
"""LLM 请求的工具调用块。= pimo.ai.types.ToolCall"""

AgentToolUpdateCallback: TypeAlias = Callable[["AgentToolResult"], None]
"""工具执行过程中用于推送增量结果的回调。

回调作用域限定于当前 ``execute()`` 调用。工具 promise 落定后的调用将被忽略。
"""

# =============================================================================
# Hook 返回值类型
# =============================================================================


@dataclass(kw_only=True)
class BeforeToolCallResult:
    """beforeToolCall hook 的返回值。

    返回 ``{block: True}`` 可阻止工具执行，Loop 将生成一条错误 tool result。
    """

    block: bool = False
    """True 时阻止工具执行。"""
    reason: str | None = None
    """阻止原因，将成为错误 tool result 的文本内容。"""


@dataclass(kw_only=True)
class AfterToolCallResult:
    """afterToolCall hook 的返回值。按字段覆盖已执行工具的结果。

    合并语义为逐字段替换:
    - ``content``: 若提供，完整替换 tool result 的 content 数组。
    - ``details``: 若提供，完整替换 tool result 的 details。
    - ``is_error``: 若提供，替换错误标记。
    - ``terminate``: 若提供，替换提前终止提示。

    未提供的字段保留原值。不执行深层合并。
    """

    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None
    terminate: bool | None = None
    """提示 Agent 应在当前工具批次后停止。
    仅当批次内所有 finalize 的工具结果均设置此标记时才提前终止。"""


# =============================================================================
# Hook 上下文类型
# =============================================================================


@dataclass(kw_only=True)
class BeforeToolCallContext:
    """传递给 beforeToolCall hook 的上下文。"""

    assistant_message: AssistantMessage
    """请求工具调用的 assistant 消息。"""
    tool_call: AgentToolCall
    """assistant_message.content 中的原始 toolCall 块。"""
    args: Any
    """经 JSON Schema 验证后的工具参数。"""
    context: AgentContext
    """工具调用准备时的当前 Agent 上下文。"""


@dataclass(kw_only=True)
class AfterToolCallContext:
    """传递给 afterToolCall hook 的上下文。"""

    assistant_message: AssistantMessage
    """请求工具调用的 assistant 消息。"""
    tool_call: AgentToolCall
    """assistant_message.content 中的原始 toolCall 块。"""
    args: Any
    """经 JSON Schema 验证后的工具参数。"""
    result: AgentToolResult
    """应用任何 afterToolCall 覆盖前的已执行工具结果。"""
    is_error: bool
    """工具结果当前是否被视为错误。"""
    context: AgentContext
    """工具调用 finalize 时的当前 Agent 上下文。"""


@dataclass(kw_only=True)
class ShouldStopAfterTurnContext:
    """传递给 shouldStopAfterTurn 回调的上下文。"""

    message: AssistantMessage
    """完成该轮的 assistant 消息。"""
    tool_results: list[ToolResultMessage]
    """传递给前置 turn_end 事件的工具结果消息。"""
    context: AgentContext
    """该轮的 assistant 消息和工具结果已追加后的当前 Agent 上下文。"""
    new_messages: list[AgentMessage]
    """本轮 loop 调用将返回的消息。
    prompt 运行包含初始提示消息；continuation 运行不包含已存在的上下文消息。"""


@dataclass(kw_only=True)
class PrepareNextTurnContext:
    """传递给 prepareNextTurn 回调的上下文。= ShouldStopAfterTurnContext"""

    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[AgentMessage]


@dataclass(kw_only=True)
class AgentLoopTurnUpdate:
    """prepareNextTurn 回调返回的替换运行时状态。

    用于在开始下一次 LLM 请求前更新上下文/模型/思考级别。
    所有字段均为可选，未提供则保持当前值。
    """

    context: AgentContext | None = None
    """下一轮 LLM 请求使用的上下文。"""
    model: Model | None = None
    """下一轮 LLM 请求使用的模型。"""
    thinking_level: ThinkingLevel | None = None
    """下一轮 LLM 请求使用的思考级别。"""


# =============================================================================
# 工具类型
# =============================================================================

_T = TypeVar("_T")
"""AgentToolResult.details 的类型变量。"""


@dataclass(kw_only=True)
class AgentToolResult(Generic[_T]):
    """工具执行的最终或部分结果。"""

    content: list[TextContent | ImageContent]
    """返回给模型的文本/图片内容。"""
    details: _T
    """供日志或 UI 渲染的结构化详情。"""
    terminate: bool = False
    """提示 Agent 应在当前工具批次后停止。
    仅当批次内所有 finalize 的工具结果均设置此标记时才提前终止。"""


@dataclass(kw_only=True)
class AgentTool(Tool):
    """Agent 运行时使用的工具定义。扩展 pimo.ai.types.Tool，增加执行逻辑。"""

    label: str
    """UI 显示的人类可读标签。"""
    prepare_arguments: Callable[[Any], dict[str, Any]] | None = None
    """可选的参数兼容层。在 JSON Schema 验证前转换原始 toolCall 参数。
    必须返回匹配 parameters schema 的 dict。"""
    execution_mode: ToolExecutionMode | None = None
    """单工具执行模式覆盖。为 None 时使用 AgentLoopConfig.toolExecution 的默认值。"""

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        """执行工具调用。

        失败时应抛出异常而非在 content 中编码错误。

        Args:
            tool_call_id: 唯一工具调用 ID。
            params: 已验证的参数。
            signal: 取消信号。实现应定期检查 ``signal.is_set()``。
            on_update: 增量结果回调。实现可在长时间执行中推送部分结果。

        Returns:
            工具执行结果。

        Raises:
            NotImplementedError: 子类必须覆盖此方法。
        """
        raise NotImplementedError(
            f"{type(self).__name__}.execute() must be implemented by subclass"
        )


# =============================================================================
# 消息联合类型
# =============================================================================

# 自定义消息类型定义在 harness/messages.py 中:
#   BashExecutionMessage, CompactionSummaryMessage, BranchSummaryMessage, CustomMessage
# 此处使用字符串前向引用避免循环导入（harness/messages.py 创建后 Pyright 错误自动消除）。
AgentMessage: TypeAlias = Union[
    Message,
    "BashExecutionMessage",          # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    "CompactionSummaryMessage",      # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    "BranchSummaryMessage",          # pyright: ignore[reportUndefinedVariable]  # noqa: F821
    "CustomMessage",                 # pyright: ignore[reportUndefinedVariable]  # noqa: F821
]
"""Agent 会话转录中可出现的消息联合类型。

= Message (标准三元组) + BashExecution + CompactionSummary + BranchSummary + Custom
其中自定义消息类型定义在 ``pimo.agent_core.harness.messages``。
"""

# =============================================================================
# Agent 上下文
# =============================================================================


@dataclass(kw_only=True)
class AgentContext:
    """传递给底层 Agent Loop 的上下文快照。"""

    system_prompt: str
    """随 LLM 请求发送的系统提示。"""
    messages: list[AgentMessage]
    """模型可见的会话转录。"""
    tools: list[AgentTool] | None = None
    """本轮可用的工具列表。为 None 表示不传递工具。"""


# =============================================================================
# Agent 状态
# =============================================================================


@dataclass(kw_only=True)
class AgentState:
    """Agent 的公开状态。

    ``tools`` 和 ``messages`` 使用 property setter 在赋值时复制顶层数组，
    防止外部意外修改内部数据。

    ``is_streaming`` / ``streaming_message`` / ``pending_tool_calls`` /
    ``error_message`` 为只读字段，由 Agent 类管理。
    """

    system_prompt: str = ""
    """随每次模型请求发送的系统提示。"""
    model: Model | None = None
    """后续轮次使用的活跃模型。"""
    thinking_level: ThinkingLevel = "off"
    """后续轮次使用的思考级别。"""
    _tools: list[AgentTool] = field(default_factory=list, repr=False)
    _messages: list[AgentMessage] = field(default_factory=list, repr=False)
    is_streaming: bool = False
    """Agent 正在处理 prompt/continuation 时为 True。"""
    streaming_message: AgentMessage | None = None
    """当前流式响应的部分 assistant 消息（如有）。"""
    pending_tool_calls: frozenset[str] = field(default_factory=frozenset)
    """当前执行中的工具调用 ID 集合。"""
    error_message: str | None = None
    """最近一次失败或中止的 assistant 轮次的错误消息（如有）。"""

    @property
    def tools(self) -> list[AgentTool]:
        """可用工具列表。赋值时复制顶层数组。"""
        return self._tools

    @tools.setter
    def tools(self, tools: list[AgentTool]) -> None:
        self._tools = list(tools)

    @property
    def messages(self) -> list[AgentMessage]:
        """会话转录。赋值时复制顶层数组。"""
        return self._messages

    @messages.setter
    def messages(self, messages: list[AgentMessage]) -> None:
        self._messages = list(messages)


# =============================================================================
# Agent 事件协议
# =============================================================================


@dataclass(kw_only=True)
class AgentStartEvent:
    """Agent 运行开始。"""
    type: Literal["agent_start"] = "agent_start"


@dataclass(kw_only=True)
class AgentEndEvent:
    """Agent 运行结束。agent_end 是 Agent 运行中最后发出的事件。"""
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage]
    """本轮产生的新消息列表。"""


@dataclass(kw_only=True)
class TurnStartEvent:
    """一轮对话开始。一轮 = 一次 assistant 响应 + 可能的多轮工具调用。"""
    type: Literal["turn_start"] = "turn_start"


@dataclass(kw_only=True)
class TurnEndEvent:
    """一轮对话结束。"""
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage
    """该轮的 assistant 消息。"""
    tool_results: list[ToolResultMessage]
    """该轮产生的工具结果消息。"""


@dataclass(kw_only=True)
class MessageStartEvent:
    """消息开始事件。对 user、assistant、toolResult 消息均触发。"""
    type: Literal["message_start"] = "message_start"
    message: AgentMessage


@dataclass(kw_only=True)
class MessageUpdateEvent:
    """消息更新事件。仅 assistant 消息在流式接收期间触发。"""
    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    """累积至今的部分 assistant 消息。"""
    assistant_message_event: AssistantMessageEvent
    """触发此更新的原始 LLM 事件。"""


@dataclass(kw_only=True)
class MessageEndEvent:
    """消息结束事件。消息完全定型后触发。"""
    type: Literal["message_end"] = "message_end"
    message: AgentMessage


@dataclass(kw_only=True)
class ToolExecutionStartEvent:
    """工具执行开始。"""
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: Any


@dataclass(kw_only=True)
class ToolExecutionUpdateEvent:
    """工具执行增量更新。"""
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any


@dataclass(kw_only=True)
class ToolExecutionEndEvent:
    """工具执行结束。"""
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool


# AgentEvent 联合类型
AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)
"""Agent 发出的所有事件联合类型，供 UI 订阅消费。

事件顺序保证:
1. agent_start → (turn_start → message_start → message_update* → message_end
   → [tool_execution_start → tool_execution_update* → tool_execution_end]* → turn_end)*
   → agent_end
"""

# =============================================================================
# Agent Loop 配置
# =============================================================================


@dataclass(kw_only=True)
class AgentLoopConfig(SimpleStreamOptions):
    """Agent Loop 的完整配置。

    继承 pimo.ai.types.SimpleStreamOptions 的所有字段（reasoning, temperature,
    max_tokens 等），并增加 Agent 专有配置。

    所有回调约定:
    - 不得抛出异常或返回 rejected promise。
    - 返回安全的回退值（如 []、undefined）代替。
    """

    model: Model
    """当前轮次使用的模型。"""

    convert_to_llm: Callable[
        [list[AgentMessage]],
        Awaitable[list[Message]],
    ]
    """将 AgentMessage[] 转换为 LLM 可理解的 Message[]。

    每条 AgentMessage 必须转换为 UserMessage、AssistantMessage 或 ToolResultMessage。
    无法转换的消息（如纯 UI 通知）应被过滤掉。

    约定: 不得抛出。返回安全的回退值代替。
    """

    # ---- 可选回调 ----

    transform_context: (
        Callable[
            [list[AgentMessage], AbortSignal | None],
            Awaitable[list[AgentMessage]],
        ]
        | None
    ) = None
    """在 convertToLlm 之前对上下文应用的变换。

    用于上下文窗口管理（裁剪旧消息）、注入外部上下文等。

    约定: 不得抛出。返回原始消息或其他安全的回退值。
    """

    get_api_key: Callable[[str], Awaitable[str | None]] | None = None
    """为每次 LLM 调用动态解析 API Key。

    用于短期 OAuth token（如 GitHub Copilot），可能在长时间工具执行中过期。

    约定: 不得抛出。无可用 key 时返回 None。
    """

    should_stop_after_turn: (
        Callable[
            [ShouldStopAfterTurnContext],
            Awaitable[bool],
        ]
        | None
    ) = None
    """每轮完全结束后调用。

    若返回 True，Loop 发出 agent_end 并在轮询 steering/follow-up 队列前退出。
    当前 assistant 响应和任何工具执行正常完成。

    约定: 不得抛出。
    """

    prepare_next_turn: (
        Callable[
            [PrepareNextTurnContext],
            Awaitable[AgentLoopTurnUpdate | None],
        ]
        | None
    ) = None
    """在 turn_end 之后、Loop 决定是否开始下一次 LLM 请求之前调用。

    返回替换的 context/model/thinking 状态以影响本轮的下一次 LLM 调用。
    返回 None 保持当前状态不变。
    """

    get_steering_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None
    """在当前 assistant 轮次完成工具执行后调用（除非 shouldStopAfterTurn
    先退出），返回要注入到对话中的 steering 消息。

    约定: 不得抛出。无消息时返回 []。
    """

    get_follow_up_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None
    """在 Agent 没有更多 tool call 也没有 steering 消息时调用。
    若返回消息，则将其加入上下文并继续下一轮。

    约定: 不得抛出。无消息时返回 []。
    """

    # ---- 工具执行配置 ----

    tool_execution: ToolExecutionMode = "parallel"
    """工具执行模式。默认 ``"parallel"``。"""

    before_tool_call: (
        Callable[
            [BeforeToolCallContext, AbortSignal | None],
            Awaitable[BeforeToolCallResult | None],
        ]
        | None
    ) = None
    """在工具执行前调用（参数已验证后）。

    返回 ``BeforeToolCallResult(block=True)`` 阻止执行。
    """

    after_tool_call: (
        Callable[
            [AfterToolCallContext, AbortSignal | None],
            Awaitable[AfterToolCallResult | None],
        ]
        | None
    ) = None
    """在工具执行完成后、tool_execution_end 和 tool-result 消息事件发出前调用。

    返回 AfterToolCallResult 覆盖已执行工具结果的字段。
    """


# =============================================================================
# 延迟导入：AssistantMessageEventStream
# =============================================================================

# AssistantMessageEventStream 通过 StreamFn 中的字符串前向引用延迟导入，
# 避免 pimo-ai → pimo-agent-core 的潜在循环导入。
# TYPE_CHECKING 块中的导入仅供静态类型检查器使用。
