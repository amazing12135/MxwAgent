"""pimo-agent-core — Agent 运行时。

提供 Agent Loop、Agent 类、AgentHarness、会话系统和上下文压缩。

公共 API:
- ``Agent`` — 有状态 Agent 封装
- ``AgentHarness`` — 编排层（会话持久化、Hook、快照）
- ``run_agent_loop()`` / ``run_agent_loop_continue()`` — 纯函数 Agent Loop
"""

from __future__ import annotations

from pimo.agent_core.types import (
    # 基础类型别名
    AbortSignal,
    AgentMessage,
    AgentToolCall,
    AgentToolUpdateCallback,
    QueueMode,
    StreamFn,
    ThinkingLevel,
    ToolExecutionMode,
    # Hook 返回值
    AfterToolCallResult,
    BeforeToolCallResult,
    # Hook 上下文
    AfterToolCallContext,
    AgentLoopTurnUpdate,
    BeforeToolCallContext,
    PrepareNextTurnContext,
    ShouldStopAfterTurnContext,
    # 工具
    AgentTool,
    AgentToolResult,
    # 状态与上下文
    AgentContext,
    AgentState,
    # 事件
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
    # 配置
    AgentLoopConfig,
)
