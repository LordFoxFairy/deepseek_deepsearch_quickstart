from typing import List, Union, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
# 移除 ToolExecutor 和 ToolNode 的导入

from backend.src.config.settings import settings
from backend.src.llms.openai_llm import get_chat_model
from backend.src.prompts.agent_decision_prompts import AGENT_DECISION_PROMPT, TOOL_FAILURE_DECISION_PROMPT
from backend.src.prompts.search_query_prompts import SEARCH_QUERY_PROMPT
from backend.src.prompts.synthesis_prompts import SYNTHESIS_PROMPT
from backend.src.schemas.graph_state import AgentState
from backend.src.tools.search_tools import search_tool
from backend.src.tools.rag_tools import rag_tool

# 初始化 LLM 和工具
llm = get_chat_model()
tools = [search_tool, rag_tool]

# 将 LLM 绑定到工具，使其能够进行工具调用
llm_with_tools = llm.bind_tools(tools)


class DeepSearchGraph:
    """
    DeepSearch 代理的 LangGraph 定义。
    这个图定义了代理如何通过一系列步骤（思考、工具调用、信息综合）来响应用户查询。
    """

    def __init__(self):
        self.workflow = StateGraph(AgentState)
        self._add_nodes()
        self._add_edges()
        self.app = self.workflow.compile()

    def _add_nodes(self):
        """
        定义 LangGraph 的各个节点。
        """
        # 代理思考节点：LLM 决定下一步行动（工具调用或最终答案）
        self.workflow.add_node("agent", self.call_agent)
        # 工具执行节点：执行代理决定的工具
        self.workflow.add_node("call_tool", self.call_tool)
        # 搜索查询生成节点：用于在需要搜索时生成精确的查询
        self.workflow.add_node("generate_search_query", self.generate_search_query)
        # 信息综合节点：综合所有研究结果生成最终答案
        self.workflow.add_node("synthesize_answer", self.synthesize_answer)

    def _add_edges(self):
        """
        定义 LangGraph 的边和条件逻辑。
        """
        # 定义入口点
        self.workflow.set_entry_point("agent")

        # 代理决策后的路由
        self.workflow.add_conditional_edges(
            "agent",
            self.route_agent_action,
            {
                "call_tool": "call_tool",
                "synthesize_answer": "synthesize_answer",
                "end": END,  # 如果代理决定结束
            },
        )

        # 工具调用后的路由
        self.workflow.add_edge("call_tool", "agent")  # 工具执行后返回代理重新思考

        # 信息综合后的路由
        self.workflow.add_edge("synthesize_answer", END)  # 综合完成后结束

    def call_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        代理思考节点：LLM 根据当前状态决定下一步行动。
        """
        print("进入 'agent' 节点: 代理正在思考...")
        # 将工具描述传递给 LLM，以便它知道如何调用工具
        # Langchain 的 with_structured_output 或 bind_tools 会自动处理工具提示
        response = llm_with_tools.invoke(
            AGENT_DECISION_PROMPT.format_messages(
                tool_names=[tool.name for tool in tools],
                current_state=state,
                input=state["input"],
                agent_scratchpad=[]  # 修正：将空字符串改为空列表，因为期望是 BaseMessage 列表
            )
        )
        # 检查 LLM 是否决定调用工具
        tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
        if tool_calls:
            print(f"代理决定调用工具: {tool_calls}")
            return {"tool_calls": tool_calls}
        else:
            print("代理决定不调用工具，可能直接给出答案或结束。")
            # 修正：即使没有工具调用，也返回一个空的 tool_calls 列表，确保状态完整性
            return {"agent_outcome": response.content, "tool_calls": []}  # 将 LLM 的内容作为代理的最终输出

    def call_tool(self, state: AgentState) -> Dict[str, Any]:
        """
        工具执行节点：执行代理决定的工具。
        """
        print("进入 'call_tool' 节点: 正在执行工具...")
        tool_calls = state["tool_calls"]
        tool_output_list = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})  # LLM 生成的工具调用通常使用 'args' 键

            # 从全局工具列表中找到对应的工具对象
            target_tool = None
            for t in tools:
                if t.name == tool_name:
                    target_tool = t
                    break

            if not target_tool:
                error_message = f"错误: 未找到名为 '{tool_name}' 的工具。"
                tool_output_list.append(error_message)
                print(error_message)
                return {
                    "tool_output": error_message,
                    "agent_outcome": f"工具执行失败：{error_message}",
                    "tool_calls": []  # 清空工具调用，防止循环
                }

            try:
                # 手动调用工具的 invoke 方法。
                # StructuredTool 期望接收一个与 args_schema 匹配的 Pydantic 对象。
                # 传入的 tool_args 字典将由 StructuredTool 内部转换为 Pydantic 对象。
                output = target_tool.invoke(tool_args)

                tool_output_list.append(f"工具 {tool_name} 输出: {output}")
                print(f"工具 '{tool_name}' 执行成功。")

                # 如果是搜索工具，将结果添加到 research_results
                if tool_name == search_tool.name:
                    state["research_results"].extend(output)
            except Exception as e:
                error_message = f"工具 '{tool_name}' 执行失败: {e}"
                tool_output_list.append(error_message)
                print(error_message)
                return {
                    "tool_output": error_message,
                    "agent_outcome": f"工具执行失败：{error_message}",
                    "tool_calls": []  # 清空工具调用，防止循环
                }

        # 将所有工具输出合并为一个字符串或列表
        return {"tool_output": "\n".join(map(str, tool_output_list))}

    def generate_search_query(self, state: AgentState) -> Dict[str, Any]:
        """
        搜索查询生成节点：根据当前状态生成搜索查询。
        """
        print("进入 'generate_search_query' 节点: 正在生成搜索查询...")
        # 使用 LLM 和 SEARCH_QUERY_PROMPT 生成搜索查询
        response = llm.invoke(
            SEARCH_QUERY_PROMPT.format_messages(
                current_state=state,
                input=state["input"]
            )
        )
        # 假设 LLM 返回的查询是多行字符串，每行一个查询
        queries = [q.strip() for q in response.content.split('\n') if q.strip()]
        print(f"生成的搜索查询: {queries}")
        # 这里需要将生成的查询转换为工具调用格式，并返回给代理
        tool_calls = []
        for q in queries:
            tool_calls.append({
                "name": search_tool.name,
                "args": {"query": q, "num_results": 5}  # 默认获取5条结果
            })
        return {"tool_calls": tool_calls}  # 返回工具调用，让代理去执行

    def synthesize_answer(self, state: AgentState) -> Dict[str, Any]:
        """
        信息综合节点：综合所有研究结果生成最终答案。
        """
        print("进入 'synthesize_answer' 节点: 正在综合答案...")
        # 格式化研究结果以便 LLM 综合
        formatted_results = []
        for res in state["research_results"]:
            # 确保 res 是字典且包含 'title', 'url', 'snippet'
            if isinstance(res, dict) and 'title' in res and 'url' in res and 'snippet' in res:
                formatted_results.append(
                    f"标题: {res['title']}\n"
                    f"URL: {res['url']}\n"
                    f"摘要: {res['snippet']}\n"
                    f"---"
                )
            elif isinstance(res, str):  # 如果是字符串形式的工具输出
                formatted_results.append(res)
            else:
                print(f"警告: 发现非预期格式的研究结果: {res}")
                formatted_results.append(str(res))

        # 使用 LLM 和 SYNTHESIS_PROMPT 综合答案
        response = llm.invoke(
            SYNTHESIS_PROMPT.format_messages(
                input=state["input"],
                research_results="\n\n".join(formatted_results)
            )
        )
        final_answer = response.content
        print(f"综合答案完成。")
        return {"final_answer": final_answer}

    def route_agent_action(self, state: AgentState) -> str:
        """
        代理决策路由函数：根据代理的输出决定下一步的图节点。
        """
        if state["tool_calls"]:
            # 简化逻辑：如果代理直接给出了工具调用，就去执行工具
            return "call_tool"

        if state["agent_outcome"]:
            # 如果代理直接给出了最终答案或结束信号
            if "无法继续" in state["agent_outcome"] or "结束" in state["agent_outcome"]:
                return "end"
            # 如果代理给出了一个答案，但没有工具调用，则认为是最终答案
            return "synthesize_answer"  # 假设 agent_outcome 包含可综合的内容

        # 如果没有明确的工具调用或最终答案，可以默认到搜索查询生成或结束
        # 这是一个需要根据实际代理行为调整的逻辑
        print("警告: 代理未给出明确的工具调用或最终答案。尝试生成搜索查询。")
        return "generate_search_query"  # 默认先尝试生成搜索查询

    def get_app(self):
        """
        返回编译后的 LangGraph 应用。
        """
        return self.app


if __name__ == "__main__":
    # 此块用于测试 DeepSearchGraph
    print("正在测试 deepsearch_graph.py...")

    # 实例化图
    deep_search_graph = DeepSearchGraph()
    app = deep_search_graph.get_app()

    # 模拟一个用户查询
    test_query = "2024年人工智能领域有哪些重要的进展和应用？"

    print(f"\n开始运行 DeepSearch 代理，查询: '{test_query}'")
    # 运行图，并迭代每一步的输出
    # 注意：这里只是一个简单的同步运行示例，实际应用中会通过 FastAPI 暴露接口
    for s in app.stream(
            {"input": test_query, "chat_history": [HumanMessage(content=test_query)], "research_results": []}):
        if "__end__" not in s:
            print(s)
            # 可以在这里根据 s 的内容更新前端显示
        else:
            print("\n--- 流程结束 ---")
            final_state = s["__end__"]
            print(f"最终答案: {final_state.get('final_answer', '无最终答案')}")
            print(f"研究结果: {final_state.get('research_results', '无研究结果')}")
            print(f"聊天历史: {final_state.get('chat_history', '无历史')}")
