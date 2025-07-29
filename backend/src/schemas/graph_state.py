import operator
from typing import TypedDict, Annotated, List, Literal, Optional, Dict, Any

from langchain_core.messages import BaseMessage


class PlanItem(TypedDict):
    """
    定义了统一计划中单个任务项（Plan Item）的完整状态。
    """
    # --- 核心标识与依赖 ---
    item_id: str
    """该计划项的唯一标识符。"""

    task_type: Literal["RESEARCH", "WRITING"]
    """
    明确该任务是“研究”类型还是“写作”类型。
    这是 supervisor 节点进行任务分派的核心依据。
    """

    description: str
    """对该计划项目标的简要描述。"""

    dependencies: List[str]
    """
    该计划项依赖的其他 PlanItem 的 item_id 列表。
    一个计划项只有在其所有依赖项都'completed'后才能开始。
    """

    # --- 核心状态 ---
    status: Literal["pending", "ready", "in_progress", "completed", "failed"]
    """
    该计划项的当前状态。
    - pending: 存在未完成的依赖项。
    - ready: 所有依赖项已完成，可以开始执行。
    - in_progress: 正在处理。
    - completed: 已成功完成。
    - failed: 任务执行失败。
    """

    # --- 内容与执行 ---
    content: str
    """
    该计划项生成的主要内容。
    - 对于 RESEARCH 任务: 存储的是研究到的原始资料或摘要。
    - 对于 WRITING 任务: 存储的是最终写好的章节内容。
    """

    summary: Optional[str]
    """对该计划项'content'的简明摘要（如果适用）。"""

    execution_log: Annotated[List[str], operator.add]
    """记录执行此计划项时的关键步骤、决策和操作日志。"""

    evaluation_results: Optional[str]
    """对该计划项当前内容的评估反馈（如果适用）。"""

    attempt_count: int
    """为完成此计划项已进行的尝试次数。"""


class AgentState(TypedDict):
    """
    高级智能体状态管理 TypedDict。
    采用统一计划，由主图集中控制，流程更清晰、高效。
    """
    input: str
    """用户的原始输入。"""

    chat_history: list[BaseMessage]
    """完整的对话历史。"""

    # --- 统一规划与大纲 ---
    overall_outline: Optional[str]
    """
    由总规划器生成的报告总大纲，用于在写作过程中保持方向一致。
    """

    plan: List[PlanItem]
    """
    统一的研究与写作计划。由总规划器生成，由主图的 supervisor 节点调度执行。
    """

    # --- 最终结果 ---
    final_answer: str
    """由最终的“整合”节点生成的完整报告。"""

    final_sources: List[Dict[str, Any]]
    """
    一个结构化的列表，用于存储最终报告的所有唯一引用来源。
    示例: [{'number': 1, 'title': 'LangGraph介绍', 'url': 'https://...'}]
    """

    # --- 运行时状态与日志 ---
    current_plan_item_id: Optional[str]
    """当前正在处理的 PlanItem 的 ID。"""

    supervisor_decision: str
    """主管代理的宏观决策，如 'RESEARCH', 'WRITING', 'PLAN' 等。"""

    step_count: int
    """当前执行的总步数，用于防止无限循环。"""

    error_log: Annotated[List[Dict[str, Any]], operator.add]
    """
    结构化的错误日志列表。
    例如: [{'node': 'research_executor', 'item_id': 'research_xyz', 'error': 'API rate limit exceeded'}]
    """

    # --- 共享上下文（保留） ---
    shared_context: Dict[str, Any]
    """
    一个全局共享的字典，主要用于管理引用。
    - 'citations': 一个用于管理全局引用的字典。
    - 'next_citation_number': 下一个可用的引用编号。
    """

    # 用于顺序执行的全局步骤索引
    next_step_index: int
