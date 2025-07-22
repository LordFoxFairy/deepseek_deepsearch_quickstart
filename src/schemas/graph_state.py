from typing import List, TypedDict, Annotated, Union, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# LangGraph 代理的增强状态定义。
# 这个状态将支持更复杂的规划、执行、评估和迭代过程。
class AgentState(TypedDict):
    """
    代表 LangGraph 代理的当前状态。
    它包含了代理在执行过程中所需的所有信息，支持多步骤、规划和评估。
    """
    input: str # 用户输入的原始查询
    chat_history: Annotated[List[BaseMessage], add_messages] # 聊天历史记录，新消息会被追加

    # 规划相关字段
    plan: List[str] # 代理为完成任务制定的步骤列表
    current_step: Optional[str] # 当前正在执行的计划步骤

    # 工具调用和执行相关字段
    tool_calls: List[dict] # 代理决定调用的工具列表 (LLM 生成的工具调用)
    tool_output: Union[str, None] # 工具执行后的输出 (原始输出)
    intermediate_steps: Annotated[List[BaseMessage], add_messages] # 代理的中间思考和工具执行日志

    # 研究结果相关字段
    research_results: List[dict] # 存储研究工具（如搜索、RAG）返回的结构化结果
    raw_research_content: str # 存储所有原始研究内容的汇总，供综合使用

    # 答案和评估相关字段
    final_answer: Union[str, None] # 代理综合后的最终答案
    evaluation_results: Union[str, None] # 对最终答案或中间结果的评估反馈
    replan_needed: bool # 指示是否需要重新规划或进一步研究

    # Supervisor 决策字段
    supervisor_decision: Optional[str] # 主管代理的决策，指示下一个要路由到的节点

    # 循环控制字段
    step_count: int # 当前执行的步骤计数，用于防止无限循环
    consecutive_no_progress_count: int # 新增：连续没有取得有效研究进展的次数

if __name__ == "__main__":
    # 此块用于测试 AgentState 的结构
    print("正在测试 graph_state.py...")
    print("AgentState 类已更新。")

    # 示例如何创建一个 AgentState 实例
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    initial_state: AgentState = {
        "input": "2024年人工智能的最新进展是什么？",
        "chat_history": [
            HumanMessage(content="你好，请问2024年人工智能的最新进展是什么？")
        ],
        "plan": [],
        "current_step": None,
        "tool_calls": [],
        "tool_output": None,
        "intermediate_steps": [],
        "research_results": [],
        "raw_research_content": "",
        "final_answer": None,
        "evaluation_results": None,
        "replan_needed": False,
        "supervisor_decision": None,
        "step_count": 0,
        "consecutive_no_progress_count": 0 # 初始化为 0
    }
    print("\n初始 AgentState 示例:")
    print(initial_state)

    # 示例如何更新状态
    updated_state = initial_state.copy()
    updated_state["chat_history"] = add_messages(updated_state["chat_history"], [AIMessage(content="好的，我正在为您查找相关信息。")])
    updated_state["plan"] = ["执行搜索", "综合结果"]
    updated_state["current_step"] = "执行搜索"
    updated_state["tool_calls"] = [{"name": "search_tool", "args": {"query": "2024 AI进展", "num_results": 3}}]
    updated_state["intermediate_steps"] = add_messages(updated_state["intermediate_steps"], [
        AIMessage(content="正在调用搜索工具..."),
        ToolMessage(content="搜索结果：...", tool_call_id="tool_call_id_1")
    ])
    updated_state["research_results"].append({"title": "AI新闻", "url": "http://example.com/ai", "snippet": "AI最新进展..."})
    updated_state["raw_research_content"] += "AI最新进展..."
    updated_state["evaluation_results"] = "需要更多信息"
    updated_state["replan_needed"] = True
    updated_state["supervisor_decision"] = "PLANNER"
    updated_state["step_count"] = 5
    updated_state["consecutive_no_progress_count"] = 2 # 模拟连续无进展

    print("\n更新后的 AgentState 示例:")
    print(updated_state)
