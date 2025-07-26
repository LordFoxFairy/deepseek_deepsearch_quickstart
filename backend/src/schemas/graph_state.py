from typing import List, TypedDict, Annotated, Union, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    定义了LangGraph代理的当前状态。
    它包含了代理在执行任务过程中所需的所有信息，以支持多步骤的规划、执行和评估。
    """
    # --- 核心输入与历史 ---
    input: str
    """用户的原始输入查询。"""

    chat_history: Annotated[List[BaseMessage], add_messages]
    """聊天历史记录，新的消息会被自动追加到列表中。"""

    # --- 规划相关字段 ---
    plan: List[str]
    """代理为完成任务而制定的步骤列表。"""

    current_step: Optional[str]
    """当前正在执行的计划步骤。"""

    planning_attempts_count: int
    """针对当前查询已经进行的规划尝试次数。"""

    # --- 工具调用与执行相关字段 ---
    tool_calls: List[dict]
    """由LLM生成的、代理决定调用的工具列表。"""

    tool_output: Union[str, None]
    """工具执行后返回的原始输出。"""

    intermediate_steps: Annotated[List[BaseMessage], add_messages]
    """代理的中间思考和工具执行日志，用于追踪过程。"""

    # --- 研究结果相关字段 ---
    research_results: List[dict]
    """存储由研究工具（如搜索、RAG）返回的结构化结果。"""

    # raw_research_content: str
    # """存储所有原始研究内容的纯文本汇总，供后续综合使用。"""

    # --- 答案与评估相关字段 ---
    final_answer: Union[str, None]
    """代理综合所有信息后生成的最终答案。"""

    evaluation_results: Union[str, None]
    """对最终答案或中间结果的评估反馈。"""

    replan_needed: bool
    """一个布尔标志，指示当前是否需要重新规划或进行进一步的研究。"""

    # --- 流程控制字段 ---
    supervisor_decision: Optional[str]
    """主管代理的决策，用于指示流程应转向哪个节点。"""

    step_count: int
    """当前任务已执行的总步骤数，用于防止无限循环。"""

    consecutive_no_progress_count: int
    """代理连续未能取得有效研究进展的步骤次数。"""
