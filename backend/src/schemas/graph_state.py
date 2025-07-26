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

    raw_research_content: str
    """存储所有原始研究内容的纯文本汇总，供后续综合使用。"""

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


from typing import List, TypedDict, Optional, Literal


class PlanItemState(TypedDict):
    """
    定义了任务计划中单个计划项（Plan Item）的完整状态。
    每个计划项代表一个需要完成的子目标或步骤，拥有独立的状态、内容和决策逻辑。
    这使得主流程可以动态地管理、评估和调整每个计划项。
    """
    # --- 核心标识与描述 ---
    item_id: str
    """该计划项的唯一标识符，例如 'research_market_trends', 'write_introduction'。"""

    description: str
    """对该计划项目标的简要描述，例如 '收集并分析2025年AI芯片市场趋势'。"""

    # --- 核心状态 (关键决策依据) ---
    status: Literal["pending", "in_progress", "completed", "needs_research", "needs_revision", "blocked"]
    """该计划项的当前状态。主流程依据此状态决定下一步动作。"""

    # --- 内容与执行 ---
    content: str
    """该计划项当前生成的主要内容或结果。例如研究摘要、草稿段落等。"""

    execution_log: List[str]
    """记录执行此计划项时的关键步骤、决策和操作日志。"""

    tool_calls: List[dict]
    """为完成此计划项而调用的工具及其参数记录。"""

    tool_output: Optional[str]
    """最近一次工具调用的原始输出。"""

    # --- 评估与反馈 ---
    evaluation_results: Optional[str]
    """对该计划项当前状态或内容的评估反馈，例如 '信息不充分，需补充最新数据'。"""

    quality_score: Optional[float]  # 可选，如果需要量化评估
    """一个0-1的分数，表示当前内容的质量或完成度。"""

    # --- 流程控制与决策 ---
    needs_research: bool
    """是否需要执行研究（如搜索、RAG）来获取更多信息。"""

    needs_revision: bool
    """是否需要基于反馈对内容进行修改或优化。"""

    requires_external_input: bool
    """是否需要用户或其他外部代理的输入才能继续。"""

    next_step_suggestion: Optional[str]
    """建议的下一步操作，例如 '调用SearchTool', '提交给用户审核', '进入下一计划项'。"""

    attempt_count: int
    """为完成此计划项已进行的尝试次数，用于防止无限循环。"""
