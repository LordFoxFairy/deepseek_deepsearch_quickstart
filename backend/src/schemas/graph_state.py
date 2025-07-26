import operator
from typing import TypedDict, Annotated, List, Literal, Optional, Dict

from langchain_core.messages import BaseMessage


class PlanItem(TypedDict):
    """
    定义了任务计划中单个计划项（Plan Item）的完整状态。
    """
    # --- 核心标识与依赖 ---
    item_id: str
    """该计划项的唯一标识符。"""

    description: str
    """对该计划项目标的简要描述。"""

    # --- 依赖管理 ---
    dependencies: List[str]
    """
    该计划项依赖的其他 PlanItem 的 item_id 列表。
    一个计划项只有在其所有依赖项都'completed'后才能开始。
    例如: 'write_section_1' 依赖于 'research_section_1_data'。
    """

    # --- 核心状态 ---
    status: Literal["pending", "ready", "in_progress", "completed", "needs_revision", "failed", "blocked"]
    """
    该计划项的当前状态。
    - pending: 存在未完成的依赖项。
    - ready: 所有依赖项已完成，可以开始执行。
    - in_progress: 正在处理。
    - completed: 已成功完成。
    - needs_revision: 需要修订。
    - failed: 尝试多次后失败。
    - blocked: 因外部原因或依赖项失败而受阻。
    """

    # --- 内容与执行 ---
    content: str
    """该计划项当前生成的主要内容或结果。"""

    execution_log: Annotated[List[str], operator.add]
    """记录执行此计划项时的关键步骤、决策和操作日志。"""

    # --- 评估与反馈 ---
    evaluation_results: Optional[str]
    """对该计划项当前内容的评估反馈。"""

    # --- 流程控制 ---
    attempt_count: int
    """为完成此计划项已进行的尝试次数。"""


class AgentState(TypedDict):
    """
    高级版智能体状态管理 TypedDict。

    参考了更成熟的 Multi-Agent 系统设计，引入了依赖管理、全局上下文和
    更丰富的错误处理机制，以支持更复杂、动态和鲁棒的任务执行流程。
    """
    input: str
    """用户的原始输入。"""

    chat_history: list[BaseMessage]
    """完整的对话历史。"""

    # --- 结构化计划 ---
    research_plan: List[PlanItem]
    """研究子图的任务计划。"""

    writing_plan: List[PlanItem]
    """写作子图的任务计划。"""

    # --- 新增: 全局上下文/草稿纸 ---
    shared_context: Dict[str, any]
    """
    一个全局共享的字典，用于存放跨节点、跨子图的关键信息、中间结论或“直觉”。
    例如: {'key_finding': 'AI市场的主要驱动力是算力需求', 'target_audience': '技术专家'}
    这比零散的日志更结构化，便于后续节点直接引用。
    """

    # --- 研究与内容 ---
    research_results: Annotated[List[dict], operator.add]
    """由 search_tool 返回的原始搜索结果列表。"""

    final_answer: str
    """由最终的“整合润色”节点生成的完整报告。"""

    # --- 流程控制与日志 ---
    current_plan_item_id: Optional[str]
    """当前正在处理的 PlanItem 的 ID。"""

    supervisor_decision: str
    """主管代理的宏观决策。"""

    intermediate_steps: Annotated[list[BaseMessage], operator.add]
    """全局的中间步骤和思考过程日志，主要用于前端展示。"""


    step_count: int
    """当前执行的总步数。"""

    # --- 新增: 错误与恢复 ---
    error_log: Annotated[List[Dict[str, str]], operator.add]
    """
    结构化的错误日志列表，记录发生的错误及其上下文。
    例如: [{'node': 'executor', 'item_id': 'research_xyz', 'error': 'API rate limit exceeded'}]
    为 supervisor 提供了更丰富的决策依据。
    """