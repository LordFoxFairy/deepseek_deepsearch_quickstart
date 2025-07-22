from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from backend.src.llms.openai_llm import get_chat_model
from backend.src.prompts.agent_decision_prompts import SUPERVISOR_DECISION_PROMPT
from backend.src.prompts.evaluator_prompts import EVALUATOR_PROMPT
from backend.src.prompts.planner_prompts import PLANNER_PROMPT
from backend.src.prompts.synthesis_prompts import SYNTHESIS_PROMPT
from backend.src.prompts.writer_prompts import WRITER_PROMPT
from backend.src.prompts.writer_reviewer_prompts import WRITER_REVIEWER_PROMPT
from backend.src.schemas.graph_state import AgentState
# 导入 SearchResult 和 RagResult 模型，以便在 synthesizer 中进行类型检查
from backend.src.schemas.tool_models import SearchResult, RagResult
from backend.src.services.llama_index_service import llama_index_service  # 导入模块级别的 llama_index_service 实例
from backend.src.tools.rag_tools import rag_tool
from backend.src.tools.search_tools import search_tool

# 初始化 LLM 和工具
llm = get_chat_model()
tools = [search_tool, rag_tool]

# 将 LLM 绑定到工具，使其能够进行工具调用
llm_with_tools = llm.bind_tools(tools)

# 定义最大步骤限制，防止无限循环
MAX_STEPS = 10  # 增加最大步骤，给代理更多尝试的机会，但仍能终止
# 定义连续无进展的阈值，超过此值则考虑终止或更激进的重新规划
NO_PROGRESS_THRESHOLD = 3


class DeepSearchGraph:
    """
    DeepSearch 代理的 LangGraph 定义，采用 Supervisor 模式。
    这个图定义了代理如何通过规划、执行、评估、写作和评审的循环来响应用户查询。
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
        # Supervisor 节点：负责路由任务给不同的子代理或结束流程
        self.workflow.add_node("supervisor", self.call_supervisor)
        # 规划器节点：生成任务执行计划
        self.workflow.add_node("planner", self.call_planner)
        # 执行器节点：执行计划中的步骤，包括工具调用
        self.workflow.add_node("executor", self.call_executor)
        # 评估器节点：评估当前研究结果或答案，决定是否需要重新规划
        self.workflow.add_node("evaluator", self.call_evaluator)
        # 写作器节点：根据研究结果撰写报告/答案
        self.workflow.add_node("writer", self.call_writer)
        # 写作评审器节点：评审写作器生成的报告/答案
        self.workflow.add_node("reviewer", self.call_reviewer)
        # 综合器节点：保留，但其功能可能被 writer 吸收或作为 writer 的前置
        self.workflow.add_node("synthesizer", self.call_synthesizer)

    def _add_edges(self):
        """
        定义 LangGraph 的边和条件逻辑。
        """
        # 定义入口点：从 supervisor 开始
        self.workflow.set_entry_point("supervisor")

        # Supervisor 决策后的路由
        self.workflow.add_conditional_edges(
            "supervisor",
            self.route_supervisor_action,
            {
                "PLANNER": "planner",
                "EXECUTOR": "executor",
                "EVALUATOR": "evaluator",
                "WRITER": "writer",
                "REVIEWER": "reviewer",
                "SYNTHESIZER": "synthesizer",
                "FINISH": END,
                "FAIL": END,
            },
        )

        # 各子代理完成后的路由：返回给 supervisor 重新决策
        self.workflow.add_edge("planner", "supervisor")  # 规划后，主管应决策执行
        self.workflow.add_edge("executor", "supervisor")  # 执行后，主管应决策评估
        self.workflow.add_edge("evaluator", "supervisor")  # 评估后，主管应根据评估结果决策

        # 写作和评审流程
        self.workflow.add_edge("writer", "reviewer")  # 写作完成后进入评审
        self.workflow.add_edge("reviewer", "supervisor")  # 评审完成后返回主管决策 (可能需要重写或结束)

        # 综合器输出后直接结束 (如果它仍被使用)
        self.workflow.add_edge("synthesizer", END)

    def call_supervisor(self, state: AgentState) -> Dict[str, Any]:
        """
        Supervisor 节点：LLM 根据当前状态决定下一步路由到哪个子代理。
        同时，在此节点增加步骤计数和最大步骤限制检查。
        """
        print("进入 'supervisor' 节点: 主管正在决策...")

        # 增加步骤计数
        state["step_count"] += 1
        print(f"当前步骤计数: {state['step_count']}")

        # 优先级最高：检查是否达到最大步骤限制
        if state["step_count"] > MAX_STEPS:
            print(f"达到最大步骤限制 ({MAX_STEPS})，强制终止。")
            return {"supervisor_decision": "FAIL", "final_answer": "任务因达到最大步骤限制而终止，未能完成。",
                    "step_count": state["step_count"]}

        # 优先级次之：检查是否连续没有取得进展
        if state["consecutive_no_progress_count"] >= NO_PROGRESS_THRESHOLD:
            print(f"警告: 连续 {state['consecutive_no_progress_count']} 步没有取得进展，强制路由到 FAIL。")
            return {"supervisor_decision": "FAIL", "final_answer": "任务因连续无进展而终止，未能完成。",
                    "step_count": state["step_count"]}

        # 将聊天历史和中间步骤作为上下文传递给主管
        messages_for_supervisor = state["chat_history"] + state["intermediate_steps"]

        response = llm.invoke(
            SUPERVISOR_DECISION_PROMPT.format_messages(
                current_state=state,
                input=state["input"],
                intermediate_steps=messages_for_supervisor
            )
        )

        # 尝试从 LLM 的响应中提取期望的关键词
        expected_decisions = ["PLANNER", "EXECUTOR", "EVALUATOR", "WRITER", "REVIEWER", "SYNTHESIZER", "FINISH", "FAIL"]

        raw_decision = response.content.strip().upper()
        supervisor_decision = "FAIL"  # 默认值，如果无法解析则失败

        for keyword in expected_decisions:
            if keyword in raw_decision:
                supervisor_decision = keyword
                break

        print(f"主管原始响应: '{response.content.strip()}'")
        print(f"解析后的主管决策: {supervisor_decision}")

        return {"supervisor_decision": supervisor_decision, "step_count": state["step_count"]}

    def call_planner(self, state: AgentState) -> Dict[str, Any]:
        """
        规划器节点：根据用户请求和当前状态生成任务计划。
        如果之前有工具失败或研究不足，规划器应该能够生成更具适应性的计划。
        """
        print("进入 'planner' 节点: 规划器正在制定计划...")

        # 收集失败和研究不足的上下文
        failure_context = ""
        if state.get("replan_needed"):  # 如果评估器标记需要重新规划
            failure_context += "注意: 上一步评估器建议重新规划。请分析原因并调整计划。\n"

        tool_failure_messages = [msg.content for msg in state["intermediate_steps"] if
                                 isinstance(msg, AIMessage) and "工具执行失败" in msg.content]
        if tool_failure_messages:
            failure_context += "之前工具执行失败的记录:\n" + "\n".join(tool_failure_messages) + "\n"

        if not state["research_results"] and state["step_count"] > 1:
            failure_context += "注意: 尽管尝试了执行，但目前没有收集到任何有效研究结果。请调整计划以确保能获取到有效信息。\n"

        # 连续无进展的上下文
        no_progress_context = ""
        if state["consecutive_no_progress_count"] > 0:
            no_progress_context = f"注意: 代理已连续 {state['consecutive_no_progress_count']} 步未能取得有效研究进展。请制定更具突破性的计划，或考虑任务的局限性。\n"

        # 确保传递给 Prompt 的 failure_context 始终是字符串
        if not failure_context:
            failure_context = "无特定失败上下文。"
        if not no_progress_context:
            no_progress_context = "无连续无进展上下文。"

        response = llm.invoke(
            PLANNER_PROMPT.format_messages(
                input=state["input"],
                current_state=state,
                failure_context=failure_context,  # 将失败上下文传递给规划器
                no_progress_context=no_progress_context  # 传递无进展上下文
            )
        )
        plan_steps = [step.strip() for step in response.content.split('\n') if step.strip()]
        print(f"生成的计划: {plan_steps}")
        # 修正：规划器完成后，明确建议主管路由到 EXECUTOR
        return {"plan": plan_steps, "current_step": plan_steps[0] if plan_steps else None,
                "supervisor_decision": "EXECUTOR"}

    def call_executor(self, state: AgentState) -> Dict[str, Any]:
        """
        执行器节点：执行计划中的当前步骤，通常涉及工具调用。
        在 search_tool 成功执行后，将搜索结果动态索引到 LlamaIndex。
        同时更新连续无进展计数。
        """
        print("进入 'executor' 节点: 执行器正在执行任务...")
        current_step = state["current_step"]
        if not current_step:
            print("错误: 没有当前步骤可执行。")
            return {"tool_output": "错误: 没有当前步骤可执行。", "replan_needed": True,
                    "supervisor_decision": "EVALUATOR"}

        print(f"执行器分析当前步骤: {current_step}")

        tool_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个工具调用专家。
            根据以下当前任务步骤，决定是否需要调用工具。
            如果需要调用工具，请以 Langchain Tools 期望的格式返回工具调用。
            可用的工具包括：{tool_names}

            当前任务步骤: {current_step}

            思考：我应该调用哪个工具来完成这个步骤？
            """),
            ("human", "请根据当前步骤生成工具调用。")
        ])

        tool_call_response = llm_with_tools.invoke(
            tool_prompt.format_messages(
                tool_names=[tool.name for tool in tools],
                current_step=current_step
            )
        )

        tool_calls = tool_call_response.tool_calls if hasattr(tool_call_response, 'tool_calls') else []

        if not tool_calls:
            print(f"执行器未能从步骤 '{current_step}' 中识别出工具调用。")
            # 如果未能识别工具调用，增加无进展计数，并路由到评估器
            return {
                "tool_output": f"步骤 '{current_step}' 已处理，但未识别到有效工具调用。",
                "replan_needed": True,
                "intermediate_steps": state["intermediate_steps"] + [
                    AIMessage(content=f"执行器未能识别工具调用: {current_step}")],
                "consecutive_no_progress_count": state["consecutive_no_progress_count"] + 1,
                "supervisor_decision": "EVALUATOR"
            }

        tool_output_list = []
        initial_research_results_count = len(state["research_results"])  # 记录执行前的研究结果数量

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})

            target_tool = None
            for t in tools:
                if t.name == tool_name:
                    target_tool = t
                    break

            if not target_tool:
                error_message = f"错误: 未找到名为 '{tool_name}' 的工具。"
                tool_output_list.append(error_message)
                print(error_message)
                # 记录错误并增加无进展计数，并路由到评估器
                return {
                    "tool_output": error_message,
                    "replan_needed": True,
                    "intermediate_steps": state["intermediate_steps"] + [
                        AIMessage(content=f"工具执行失败：{error_message}. 代理需要重新评估。")],
                    "consecutive_no_progress_count": state["consecutive_no_progress_count"] + 1,
                    "supervisor_decision": "EVALUATOR"
                }

            try:
                output = target_tool.invoke(tool_args)

                tool_output_list.append(f"工具 '{tool_name}' 输出: {output}")
                print(f"工具 '{tool_name}' 执行成功。")

                if tool_name == search_tool.name:
                    state["research_results"].extend(output)
                    for res in output:
                        if isinstance(res, dict) and 'snippet' in res:
                            state["raw_research_content"] += res['snippet'] + "\n"

                    # 关键新增：将搜索结果动态索引到 LlamaIndex
                    print(f"将 {len(output)} 条搜索结果添加到 LlamaIndex...")
                    try:
                        llama_index_service.add_search_results_to_index(output)
                        print("搜索结果已添加到 LlamaIndex。")
                    except Exception as e:
                        # 捕获 DashScope 连接错误，并将其视为工具执行失败
                        index_error_message = f"LlamaIndex 索引搜索结果失败: {e}"
                        print(index_error_message)
                        return {
                            "tool_output": index_error_message,
                            "replan_needed": True,
                            "intermediate_steps": state["intermediate_steps"] + [
                                AIMessage(content=f"工具执行失败：{index_error_message}. 代理需要重新评估。")],
                            "consecutive_no_progress_count": state["consecutive_no_progress_count"] + 1,
                            "supervisor_decision": "EVALUATOR"
                        }

                elif tool_name == rag_tool.name:
                    state["research_results"].extend(output)
                    for res in output:
                        if isinstance(res, dict) and 'content' in res:
                            state["raw_research_content"] += res['content'] + "\n"

            except Exception as e:
                error_message = f"工具 '{tool_name}' 执行失败: {e}"
                tool_output_list.append(error_message)
                print(error_message)
                return {
                    "tool_output": error_message,
                    "replan_needed": True,
                    "intermediate_steps": state["intermediate_steps"] + [
                        AIMessage(content=f"工具执行失败：{error_message}. 代理需要重新评估。")],
                    "consecutive_no_progress_count": state["consecutive_no_progress_count"] + 1,
                    "supervisor_decision": "EVALUATOR"
                }

        # 检查是否有新的研究结果被添加，如果没有则增加无进展计数
        if len(state["research_results"]) == initial_research_results_count:
            state["consecutive_no_progress_count"] += 1
            print(f"执行器未取得新研究进展，连续无进展计数: {state['consecutive_no_progress_count']}")
        else:
            state["consecutive_no_progress_count"] = 0  # 取得进展，重置计数
            print("执行器取得新研究进展，重置连续无进展计数。")

        # 推进到计划的下一个步骤
        if state["plan"] and current_step in state["plan"]:
            next_step_index = state["plan"].index(current_step) + 1
            next_step = state["plan"][next_step_index] if next_step_index < len(state["plan"]) else None
        else:
            next_step = None

        executor_log_content = f"执行器完成步骤: {current_step}. 工具输出: {''.join(tool_output_list)}"

        # 修正：执行器完成后，明确建议主管路由到 EVALUATOR
        return {
            "tool_output": "\n".join(map(str, tool_output_list)),
            "current_step": next_step,
            "intermediate_steps": state["intermediate_steps"] + [AIMessage(content=executor_log_content)],
            "consecutive_no_progress_count": state["consecutive_no_progress_count"],  # 确保更新后的计数被传递
            "supervisor_decision": "EVALUATOR"
        }

    def call_evaluator(self, state: AgentState) -> Dict[str, Any]:
        """
        评估器节点：评估当前研究结果或答案，决定是否需要重新规划。
        更智能地判断何时需要重新规划，何时应该尝试其他路径，或者何时应该直接报告无法完成。
        """
        print("进入 'evaluator' 节点: 评估器正在评估...")

        # 收集评估上下文
        evaluation_context = ""
        tool_failure_in_history = False
        for msg in state["intermediate_steps"]:
            if isinstance(msg, AIMessage) and "工具执行失败" in msg.content:
                tool_failure_in_history = True
                evaluation_context += f"历史工具失败: {msg.content}\n"

        has_research_results = bool(state["research_results"])
        has_final_answer = bool(state["final_answer"])

        if not has_research_results and state["step_count"] > 1:
            evaluation_context += "注意: 尚未收集到有效的研究结果。这可能需要调整搜索策略或尝试其他方法。\n"

        if tool_failure_in_history and not has_research_results and state["step_count"] >= MAX_STEPS / 2:
            evaluation_context += f"警告: 经过 {state['step_count']} 步尝试，工具持续失败且没有研究结果。考虑任务是否可完成。\n"

        # 连续无进展的上下文
        no_progress_context = ""
        if state["consecutive_no_progress_count"] >= NO_PROGRESS_THRESHOLD:
            no_progress_context = f"警告: 代理已连续 {state['consecutive_no_progress_count']} 步未能取得有效研究进展。这可能表明任务难以完成或需要彻底改变策略。\n"

        response = llm.invoke(
            EVALUATOR_PROMPT.format_messages(
                input=state["input"],
                current_state=state,
                research_results=state["research_results"],
                final_answer=state["final_answer"],
                evaluation_context=evaluation_context,  # 传递评估上下文
                no_progress_context=no_progress_context  # 传递无进展上下文
            )
        )
        evaluation_result = response.content.strip()
        print(f"评估结果: {evaluation_result}")

        # 修正：评估器根据 LLM 输出的单词建议，直接设置 supervisor_decision
        suggested_supervisor_decision = evaluation_result.upper()  # 假设 LLM 只输出一个单词，并将其转为大写

        # 额外的内部逻辑来设置 replan_needed，供路由函数使用
        replan_needed = False
        if suggested_supervisor_decision == "PLANNER":
            replan_needed = True

        print(f"评估器建议主管决策: {suggested_supervisor_decision}")

        return {
            "evaluation_results": evaluation_result,
            "replan_needed": replan_needed,  # 这个字段仍然有用，用于 route_supervisor_action 的强制路由
            "intermediate_steps": state["intermediate_steps"] + [
                AIMessage(content=f"评估器完成评估. 结果: {evaluation_result}")],
            "supervisor_decision": suggested_supervisor_decision  # 评估器明确建议主管下一步的路由
        }

    def call_writer(self, state: AgentState) -> Dict[str, Any]:
        """
        写作器节点：根据所有研究结果撰写报告/答案。
        """
        print("进入 'writer' 节点: 写作器正在撰写报告...")
        response = llm.invoke(
            WRITER_PROMPT.format_messages(
                input=state["input"],
                raw_research_content=state["raw_research_content"],
                research_results=state["research_results"]
            )
        )
        generated_answer = response.content
        print("写作器完成报告撰写。")
        # 修正：写作器完成后，建议主管路由到 REVIEWER
        return {
            "final_answer": generated_answer,
            "intermediate_steps": state["intermediate_steps"] + [AIMessage(content="写作器完成报告撰写。")],
            "supervisor_decision": "REVIEWER"
        }

    def call_reviewer(self, state: AgentState) -> Dict[str, Any]:
        """
        写作评审器节点：评审写作器生成的报告，并决定是否需要修改。
        """
        print("进入 'reviewer' 节点: 评审器正在评审报告...")
        response = llm.invoke(
            WRITER_REVIEWER_PROMPT.format_messages(
                input=state["input"],
                research_results=state["research_results"],
                generated_answer=state["final_answer"]
            )
        )
        review_result = response.content.strip()
        print(f"评审结果: {review_result}")

        replan_needed = False
        suggested_supervisor_decision = "FINISH"  # 默认评审通过，直接 FINISH

        if "需要修改" in review_result:
            replan_needed = True
            suggested_supervisor_decision = "PLANNER"  # 建议路由到规划器
            print("评审器建议修改报告，将返回主管重新规划。")
        else:
            print("评审通过，报告可以直接提交。")

        # 修正：评审器完成后，明确建议主管下一步的路由
        return {
            "evaluation_results": review_result,
            "replan_needed": replan_needed,
            "intermediate_steps": state["intermediate_steps"] + [
                AIMessage(content=f"评审器完成评审. 结果: {review_result}")],
            "supervisor_decision": suggested_supervisor_decision
        }

    def call_synthesizer(self, state: AgentState) -> Dict[str, Any]:
        """
        综合器节点：综合所有研究结果生成最终答案。
        注意：在引入 writer 节点后，此节点可能变得冗余，或者用于在 writer 失败时的备用方案。
        """
        print("进入 'synthesizer' 节点: 正在综合答案...")
        formatted_results = []
        for res in state["research_results"]:
            # 修正：直接检查 Pydantic 模型实例类型
            if isinstance(res, SearchResult):
                formatted_results.append(
                    f"标题: {res.title}\nURL: {res.url}\n摘要: {res.snippet}\n---"
                )
            elif isinstance(res, RagResult):
                formatted_results.append(
                    f"内容: {res.content}\n来源: {res.source}\n---"
                )
            elif isinstance(res, str):
                formatted_results.append(res)
            else:
                print(f"警告: 发现非预期格式的研究结果: {res}")
                formatted_results.append(str(res))

        response = llm.invoke(
            SYNTHESIS_PROMPT.format_messages(
                input=state["input"],
                research_results="\n\n".join(formatted_results)
            )
        )
        final_answer = response.content
        print(f"综合答案完成。")
        # 修正：综合器完成后，建议主管路由到 FINISH
        return {
            "final_answer": final_answer,
            "intermediate_steps": state["intermediate_steps"] + [AIMessage(content=f"综合器完成答案生成.")],
            "supervisor_decision": "FINISH"
        }

    def route_supervisor_action(self, state: AgentState) -> str:
        """
        Supervisor 路由函数：根据主管的输出决定下一步的图节点。
        """
        supervisor_decision = state.get("supervisor_decision", "").strip("'\"")

        # 优先级最高：如果连续没有取得进展，且达到阈值，则强制路由到 FAIL
        if state["consecutive_no_progress_count"] >= NO_PROGRESS_THRESHOLD:
            print(f"路由函数：警告: 连续 {state['consecutive_no_progress_count']} 步没有取得进展，强制路由到 FAIL。")
            return "FAIL"

        # 优先级：如果评审器建议重新规划，则强制路由到 PLANNER
        if state.get("replan_needed"):
            print("路由函数：评审器建议重新规划，强制路由到 PLANNER。")
            state["replan_needed"] = False
            return "PLANNER"

        # 优先级：如果研究结果为空且步骤计数较高，且多次尝试后仍无进展，考虑 FAIL
        if not state["research_results"] and state["step_count"] >= MAX_STEPS - 1:
            print("路由函数：警告: 接近最大步骤限制但无研究结果，强制路由到 FAIL。")
            return "FAIL"

        # 优先级：如果有最终答案，且没有 replan_needed，则路由到 FINISH
        if state["final_answer"] and not state.get("replan_needed"):
            print("路由函数：检测到最终答案，且无需重新规划，路由到 FINISH。")
            return "FINISH"

        # 正常路由逻辑：根据子代理的建议或 LLM 的决策进行路由
        if supervisor_decision == "PLANNER":
            return "PLANNER"
        elif supervisor_decision == "EXECUTOR":
            return "EXECUTOR"
        elif supervisor_decision == "EVALUATOR":
            return "EVALUATOR"
        elif supervisor_decision == "WRITER":
            return "WRITER"
        elif supervisor_decision == "REVIEWER":
            return "REVIEWER"
        elif supervisor_decision == "SYNTHESIZER":
            return "SYNTHESIZER"
        elif supervisor_decision == "FINISH":
            return "FINISH"
        elif supervisor_decision == "FAIL":
            return "FAIL"
        else:
            print(f"路由函数：主管决策 '{supervisor_decision}' 不明确，默认路由到规划器。")
            return "PLANNER"

    def get_app(self):
        """
        返回编译后的 LangGraph 应用。
        """
        return self.app


if __name__ == "__main__":
    # 此块用于测试 DeepSearchGraph
    print("正在测试 deepsearch_graph.py...")

    deep_search_graph = DeepSearchGraph()
    app = deep_search_graph.get_app()

    test_query = "2024年人工智能领域有哪些重要的进展和应用？"

    print(f"\n开始运行 DeepSearch 代理，查询: '{test_query}'")
    for s in app.stream({
        "input": test_query,
        "chat_history": [HumanMessage(content=test_query)],
        "research_results": [],
        "intermediate_steps": [],
        "plan": [],
        "current_step": None,
        "tool_calls": [],
        "tool_output": None,
        "raw_research_content": "",
        "final_answer": None,
        "evaluation_results": None,
        "replan_needed": False,
        "supervisor_decision": None,
        "step_count": 0,
        "consecutive_no_progress_count": 0  # 初始化连续无进展计数
    }):
        if "__end__" not in s:
            print(s)
        else:
            print("\n--- 流程结束 ---")
            final_state = s["__end__"]
            print(f"最终答案: {final_state.get('final_answer', '无最终答案')}")
            print(f"研究结果: {final_state.get('research_results', '无研究结果')}")
            print(f"聊天历史: {final_state.get('chat_history', '无历史')}")
            print(f"最终中间步骤: {final_state.get('intermediate_steps', '无中间步骤')}")
