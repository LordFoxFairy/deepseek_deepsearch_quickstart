from typing import List, TypedDict, Annotated, Union

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# LangGraph 代理的状态定义。
# TypedDict 用于定义字典的结构和类型，使其更具可读性和类型安全性。
# Annotated 用于在 TypedDict 中为列表类型添加合并行为（add_messages）。
class AgentState(TypedDict):
    """
    代表 LangGraph 代理的当前状态。
    它包含了代理在执行过程中所需的所有信息。
    """
    input: str  # 用户输入的原始查询
    chat_history: Annotated[List[BaseMessage], add_messages]  # 聊天历史记录，新消息会被追加
    agent_outcome: Union[str, None]  # 代理的最终决策或输出
    tool_calls: List[dict]  # 代理决定调用的工具列表
    tool_output: Union[str, None]  # 工具执行后的输出
    research_results: List[dict]  # 存储研究工具（如搜索、RAG）返回的结果
    final_answer: Union[str, None]  # 代理综合后的最终答案
