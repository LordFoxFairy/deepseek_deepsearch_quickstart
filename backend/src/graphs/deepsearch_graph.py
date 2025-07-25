from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from backend.src.llms.openai_llm import get_chat_model
from backend.src.prompts.agent_decision_prompts import SUPERVISOR_DECISION_PROMPT
from backend.src.prompts.evaluator_prompts import EVALUATOR_PROMPT
from backend.src.prompts.planner_prompts import OUTLINE_PLANNER_PROMPT, WRITER_PLANNER_PROMPT
from backend.src.prompts.synthesis_prompts import SYNTHESIS_PROMPT
from backend.src.prompts.writer_prompts import WRITER_PROMPT
from backend.src.prompts.writer_reviewer_prompts import WRITER_REVIEWER_PROMPT
from backend.src.schemas.graph_state import AgentState
from backend.src.schemas.tool_models import SearchResult, RagResult
from backend.src.services.llama_index_service import llama_index_service
from backend.src.tools.rag_tools import rag_tool
from backend.src.tools.search_tools import search_tool

# 为代理初始化LLM和工具
llm = get_chat_model()
tools = [search_tool, rag_tool]

# 将LLM与可用工具绑定，使其能够进行工具调用
llm_with_tools = llm.bind_tools(tools)

# --- 代理配置 ---
# 最大步骤数，以防止无限循环
MAX_STEPS = 15
# 连续未取得进展的步骤阈值，超过该阈值将导致任务终止
NO_PROGRESS_THRESHOLD = 3
# 规划尝试的阈值，超过后评估器应在有数据的情况下强制进入写作流程
PLANNING_ATTEMPTS_THRESHOLD = 3


class DeepSearchGraph:
    """
    使用主管（Supervisor）模式定义DeepSearch代理的图结构。
    该图协调代理的工作流程，通过规划、执行、评估、写作和审查等阶段来响应用户查询。
    """

    def __init__(self):
        self.workflow = StateGraph(AgentState)
        self._add_nodes()
        self._add_edges()
        self.app = self.workflow.compile()

    def _add_nodes(self):
        """
        将所有节点添加到工作流图中。每个节点代表代理流程中的一个特定功能或阶段。
        """
        self.workflow.add_node("supervisor", self.call_supervisor)
        self.workflow.add_node("outline_planner", self.call_outline_planner)
        self.workflow.add_node("writer_planner", self.call_writer_planner)
        self.workflow.add_node("executor", self.call_executor)
        self.workflow.add_node("evaluator", self.call_evaluator)
        self.workflow.add_node("writer", self.call_writer)
        self.workflow.add_node("reviewer", self.call_reviewer)
        self.workflow.add_node("synthesizer", self.call_synthesizer)

    def _add_edges(self):
        """
        定义工作流图中节点之间的连接和条件逻辑。
        """
        self.workflow.set_entry_point("supervisor")

        # 基于主管决策的条件路由
        self.workflow.add_conditional_edges(
            "supervisor",
            self.route_supervisor_action,
            {
                "OUTLINE_PLANNER": "outline_planner",
                "WRITER_PLANNER": "writer_planner",
                "EXECUTOR": "executor",
                "EVALUATOR": "evaluator",
                "WRITER": "writer",
                "REVIEWER": "reviewer",
                "SYNTHESIZER": "synthesizer",
                "FINISH": END,
                "FAIL": END,
            },
        )

        # 各子代理完成后返回主管以进行下一步决策的边
        self.workflow.add_edge("outline_planner", "supervisor")
        self.workflow.add_edge("writer_planner", "supervisor")
        self.workflow.add_edge("executor", "supervisor")
        self.workflow.add_edge("evaluator", "supervisor")
        self.workflow.add_edge("reviewer", "supervisor")

        # 写作和审查子流程的边
        self.workflow.add_edge("writer", "reviewer")

        # 合成器节点（如果使用）会终止图
        self.workflow.add_edge("synthesizer", END)

    def call_supervisor(self, state: AgentState) -> Dict[str, Any]:
        """
        主管节点：根据当前状态，由LLM决定到下一个子代理的路由。
        同时，此处会增加步骤计数并检查最大步骤限制。
        """
        print(f"\n--- [Step {state['step_count']}] ---")
        print("Entering 'supervisor' node: Supervisor is making a decision...")

        state["step_count"] += 1
        print(f"Current step count: {state['step_count']}")

        # 最高优先级：检查是否达到最大步骤限制
        if state["step_count"] > MAX_STEPS:
            print(f"Maximum step limit ({MAX_STEPS}) reached, forcing termination.")
            return {"supervisor_decision": "FAIL", "final_answer": "任务因达到最大步骤限制而终止，未能完成。",
                    "step_count": state["step_count"]}

        # 次高优先级：检查是否连续缺乏进展
        if state["consecutive_no_progress_count"] >= NO_PROGRESS_THRESHOLD:
            print(
                f"Warning: Consecutive {state['consecutive_no_progress_count']} steps without progress, forcing route to FAIL.")
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

        # 尝试从LLM的响应中提取预期的决策关键词
        expected_decisions = ["OUTLINE_PLANNER", "WRITER_PLANNER", "EXECUTOR", "EVALUATOR", "WRITER", "REVIEWER", "SYNTHESIZER", "FINISH", "FAIL"]

        raw_decision = response.content.strip().upper()
        supervisor_decision = "FAIL"  # 如果无法解析，则默认为FAIL

        for keyword in expected_decisions:
            if keyword in raw_decision:
                supervisor_decision = keyword
                break

        print(f"Supervisor raw response: '{raw_decision}'")
        print(f"Parsed supervisor decision: {supervisor_decision}")

        return {"supervisor_decision": supervisor_decision, "step_count": state["step_count"]}

    def call_outline_planner(self, state: AgentState) -> Dict[str, Any]:
        """
        规划器节点：根据用户请求和当前状态生成任务计划。
        如果存在先前的失败或研究不足，规划器应调整其策略。
        """
        print("Entering 'outline_planner' node: Planner is formulating a plan...")

        state["planning_attempts_count"] += 1
        print(f"Current planning attempts: {state['planning_attempts_count']}")

        # 收集有关失败和研究不足的上下文
        failure_context = ""
        if state.get("replan_needed"):
            failure_context += "注意：先前的评估器建议重新规划。请分析原因并调整计划。\n"

        tool_failure_messages = [msg.content for msg in state["intermediate_steps"] if
                                 isinstance(msg, AIMessage) and "工具执行失败" in msg.content]
        if tool_failure_messages:
            failure_context += "先前的工具执行失败记录：\n" + "\n".join(tool_failure_messages) + "\n"

        if not state["research_results"] and state["step_count"] > 1:
            failure_context += "注意：尽管尝试了执行，但尚未收集到有效的研究结果。请调整计划以确保获取有效信息。\n"

        # 关于连续无进展的上下文
        no_progress_context = ""
        if state["consecutive_no_progress_count"] > 0:
            no_progress_context = f"注意：代理已连续 {state['consecutive_no_progress_count']} 步未能取得有效的研究进展。请制定一个更具突破性的计划，或考虑任务的局限性。\n"

        response = llm.invoke(
            OUTLINE_PLANNER_PROMPT.format_messages(
                query=state["input"],
                current_state=state,
                failure_context=failure_context or "没有特定的失败上下文。",
                no_progress_context=no_progress_context or "没有连续无进展的上下文。"
            )
        )
        plan_steps = [step.strip() for step in response.content.split('\n') if step.strip()]
        print(f"Generated plan: {plan_steps}")
        return {"plan": plan_steps, "search_queries": plan_steps, "current_step": plan_steps[0] if plan_steps else None,
                "supervisor_decision": "EXECUTOR", "planning_attempts_count": state["planning_attempts_count"]}

    def call_writer_planner(self, state: AgentState) -> Dict[str, Any]:
        """
        写作规划器节点：根据已收集的研究结果，为最终报告制定一个详细的写作大纲。
        """
        print("Entering 'writer_planner' node: Writer Planner is formulating a writing plan...")

        response = llm.invoke(
            WRITER_PLANNER_PROMPT.format_messages(
                query=state["input"],
                search_results=state["research_results"]
            )
        )
        writing_plan_steps = [step.strip() for step in response.content.split('\n') if step.strip()]
        print(f"Generated writing plan: {writing_plan_steps}")

        return {
            "plan": writing_plan_steps,
            "intermediate_steps": state["intermediate_steps"] + [
                AIMessage(content=f"写作规划器完成规划。计划：{writing_plan_steps}")],
            "supervisor_decision": "WRITER"
        }

    def call_executor(self, state: AgentState) -> Dict[str, Any]:
        """
        执行器节点：执行计划中的当前步骤，通常涉及工具调用。
        在search_tool成功执行后，动态地将搜索结果索引到LlamaIndex中。
        同时更新连续无进展的计数。
        """
        print("Entering 'executor' node: Executor is performing the task...")
        current_step = state["current_step"]
        if not current_step:
            print("Error: No current step to execute.")
            return {"tool_output": "错误：没有要执行的当前步骤。", "replan_needed": True,
                    "supervisor_decision": "EVALUATOR"}

        print(f"Executor analyzing current step: {current_step}")

        tool_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个工具调用专家。
            根据下面的当前任务步骤，决定是否需要调用工具。
            如果需要调用工具，请以Langchain Tools期望的格式返回工具调用。
            可用工具包括：{tool_names}

            当前任务步骤：{current_step}

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
            return {
                "tool_output": f"步骤 '{current_step}' 已处理，但未识别出有效的工具调用。",
                "replan_needed": True,
                "intermediate_steps": state["intermediate_steps"] + [
                    AIMessage(content=f"执行器未能识别工具调用：{current_step}")],
                "consecutive_no_progress_count": state["consecutive_no_progress_count"] + 1,
                "supervisor_decision": "EVALUATOR"
            }

        tool_output_list = []
        initial_research_results_count = len(state["research_results"])

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})

            target_tool = None
            for t in tools:
                if t.name == tool_name:
                    target_tool = t
                    break

            if not target_tool:
                error_message = f"错误：未找到名为 '{tool_name}' 的工具。"
                tool_output_list.append(error_message)
                print(error_message)
                return {
                    "tool_output": error_message,
                    "replan_needed": True,
                    "intermediate_steps": state["intermediate_steps"] + [
                        AIMessage(content=f"工具执行失败：{error_message}。代理需要重新评估。")],
                    "consecutive_no_progress_count": state["consecutive_no_progress_count"] + 1,
                    "supervisor_decision": "EVALUATOR"
                }

            try:
                output = target_tool.invoke(tool_args)

                tool_output_list.append(f"工具 '{tool_name}' 输出：{output}")
                print(f"工具 '{tool_name}' 已成功执行。")

                if tool_name == search_tool.name:
                    state["research_results"].extend(output)
                    for res in output:
                        if isinstance(res, dict) and 'snippet' in res:
                            state["raw_research_content"] += res['snippet'] + "\n"

                    # 动态地将搜索结果索引到LlamaIndex
                    print(f"正在向LlamaIndex添加 {len(output)} 条搜索结果...")
                    try:
                        llama_index_service.add_search_results_to_index(output)
                        print("搜索结果已添加到LlamaIndex。")
                    except Exception as e:
                        # 捕获连接错误并将其视为工具执行失败
                        index_error_message = f"LlamaIndex对搜索结果的索引失败：{e}"
                        print(index_error_message)
                        return {
                            "tool_output": index_error_message,
                            "replan_needed": True,
                            "intermediate_steps": state["intermediate_steps"] + [
                                AIMessage(content=f"工具执行失败：{index_error_message}。代理需要重新评估。")],
                            "consecutive_no_progress_count": state["consecutive_no_progress_count"] + 1,
                            "supervisor_decision": "EVALUATOR"
                        }

                elif tool_name == rag_tool.name:
                    state["research_results"].extend(output)
                    for res in output:
                        if isinstance(res, dict) and 'content' in res:
                            state["raw_research_content"] += res['content'] + "\n"

            except Exception as e:
                error_message = f"工具 '{tool_name}' 执行失败：{e}"
                tool_output_list.append(error_message)
                print(error_message)
                return {
                    "tool_output": error_message,
                    "replan_needed": True,
                    "intermediate_steps": state["intermediate_steps"] + [
                        AIMessage(content=f"工具执行失败：{error_message}。代理需要重新评估。")],
                    "consecutive_no_progress_count": state["consecutive_no_progress_count"] + 1,
                    "supervisor_decision": "EVALUATOR"
                }

        # 检查是否添加了新的研究结果，如果没有，则增加无进展计数
        if len(state["research_results"]) == initial_research_results_count:
            state["consecutive_no_progress_count"] += 1
            print(f"执行器未取得新的研究进展，连续无进展计数：{state['consecutive_no_progress_count']}")
        else:
            state["consecutive_no_progress_count"] = 0  # 取得了进展，重置计数
            print("执行器取得了新的研究进展，重置连续无进展计数。")

        # 前进到计划中的下一步
        if state["plan"] and current_step in state["plan"]:
            next_step_index = state["plan"].index(current_step) + 1
            next_step = state["plan"][next_step_index] if next_step_index < len(state["plan"]) else None
        else:
            next_step = None

        executor_log_content = f"执行器完成步骤：{current_step}。工具输出：{''.join(tool_output_list)}"

        return {
            "tool_output": "\n".join(map(str, tool_output_list)),
            "current_step": next_step,
            "intermediate_steps": state["intermediate_steps"] + [AIMessage(content=executor_log_content)],
            "consecutive_no_progress_count": state["consecutive_no_progress_count"],
            "supervisor_decision": "EVALUATOR"
        }

    def call_evaluator(self, state: AgentState) -> Dict[str, Any]:
        """
        评估器节点：评估当前的研究结果或答案，决定是否需要重新规划。
        更智能地决定何时重新规划、何时尝试其他路径，或何时报告无法完成。
        """
        print("Entering 'evaluator' node: Evaluator is assessing...")

        # 收集评估上下文
        evaluation_context = ""
        tool_failure_in_history = False
        for msg in state["intermediate_steps"]:
            if isinstance(msg, AIMessage) and "工具执行失败" in msg.content:
                tool_failure_in_history = True
                evaluation_context += f"历史工具失败记录：{msg.content}\n"

        has_research_results = bool(state["research_results"])
        has_final_answer = bool(state["final_answer"])

        if not has_research_results and state["step_count"] > 1:
            evaluation_context += "注意：尚未收集到有效的研究结果。这可能需要调整搜索策略或尝试其他方法。\n"

        if tool_failure_in_history and not has_research_results and state["step_count"] >= MAX_STEPS / 2:
            evaluation_context += f"警告：经过 {state['step_count']} 次尝试后，工具仍然失败且没有研究结果。请考虑任务是否可完成。\n"

        # 关于连续无进展的上下文
        no_progress_context = ""
        if state["consecutive_no_progress_count"] >= NO_PROGRESS_THRESHOLD:
            no_progress_context = f"警告：代理已连续 {state['consecutive_no_progress_count']} 步未能取得有效的研究进展。这可能表明任务难以完成或需要彻底改变策略。\n"

        response = llm.invoke(
            EVALUATOR_PROMPT.format_messages(
                input=state["input"],
                current_state=state,
                research_results=state["research_results"],
                final_answer=state["final_answer"],
                evaluation_context=evaluation_context,
                no_progress_context=no_progress_context
            )
        )
        evaluation_result = response.content.strip()
        print(f"Evaluation result: {evaluation_result}")

        llm_decision = evaluation_result.upper()
        replan_needed = False

        # 1. 优先处理失败条件
        if "FAIL" in llm_decision or state["consecutive_no_progress_count"] >= NO_PROGRESS_THRESHOLD:
            print("评估器：由于关键条件（LLM建议或连续无进展），强制路由到FAIL。")
            suggested_supervisor_decision = "FAIL"
            # 2. 如果研究结果足够，进入写作规划阶段
        elif "WRITER" in llm_decision:  # 对应 prompt 中的 WRITER 建议
            print("评估器：研究结果足够，强制路由到WRITER_PLANNER。")
            suggested_supervisor_decision = "WRITER_PLANNER"
            # 3. 如果规划次数过多且有结果，也强制进入写作，避免死循环
        elif has_research_results and state["planning_attempts_count"] >= PLANNING_ATTEMPTS_THRESHOLD:
            print(f"评估器：有研究结果且规划尝试次数 >= {PLANNING_ATTEMPTS_THRESHOLD}，强制路由到WRITER_PLANNER。")
            suggested_supervisor_decision = "WRITER_PLANNER"
            # 4. 默认情况是重新规划
        else:  # 对应 prompt 中的 PLANNER 建议
            print(f"评估器：没有足够的研究结果或LLM建议重新规划，路由到OUTLINE_PLANNER。")
            replan_needed = True
            suggested_supervisor_decision = "OUTLINE_PLANNER"

        return {
            "evaluation_results": evaluation_result,
            "replan_needed": replan_needed,
            "intermediate_steps": state["intermediate_steps"] + [
                AIMessage(content=f"评估器完成评估。结果：{evaluation_result}")],
            "supervisor_decision": suggested_supervisor_decision
        }

    def call_writer(self, state: AgentState) -> Dict[str, Any]:
        """
        写作者节点：根据所有研究结果撰写报告/答案。
        """
        print("Entering 'writer' node: Writer is drafting the report...")
        response = llm.invoke(
            WRITER_PROMPT.format_messages(
                input=state["input"],
                plan=state["plan"],
                raw_research_content=state["raw_research_content"],
                research_results=state["research_results"]
            )
        )
        generated_answer = response.content
        print("Writer completed report drafting.")
        return {
            "final_answer": generated_answer,
            "intermediate_steps": state["intermediate_steps"] + [AIMessage(content="写作者完成报告起草。")],
            "supervisor_decision": "REVIEWER"
        }

    def call_reviewer(self, state: AgentState) -> Dict[str, Any]:
        """
        审阅者节点：审阅写作者生成的报告，并决定是否需要修改。
        """
        print("Entering 'reviewer' node: Reviewer is reviewing the report...")
        response = llm.invoke(
            WRITER_REVIEWER_PROMPT.format_messages(
                input=state["input"],
                research_results=state["research_results"],
                generated_answer=state["final_answer"]
            )
        )
        review_result = response.content.strip()
        print(f"Review result: {review_result}")

        if "OUTLINE_PLANNER" in review_result.upper():
            replan_needed = True
            suggested_supervisor_decision = "OUTLINE_PLANNER"
            print("审阅者认为内容不足，返回主管进行重新研究。")
        elif "WRITER_PLANNER" in review_result.upper():
            # 注意：这里 replan_needed 必须为 False，以避免被路由器的 replan 逻辑覆盖
            replan_needed = False
            suggested_supervisor_decision = "WRITER_PLANNER"
            print("审阅者认为结构有问题，返回主管调整写作大纲。")
        else: # 默认或包含“FINISH”
            replan_needed = False
            suggested_supervisor_decision = "FINISH"
            print("审阅通过，报告可以直接提交。")

        return {
            "evaluation_results": review_result,
            "replan_needed": replan_needed,
            "intermediate_steps": state["intermediate_steps"] + [
                AIMessage(content=f"审阅者完成审阅。结果：{review_result}")],
            "supervisor_decision": suggested_supervisor_decision
        }

    def call_synthesizer(self, state: AgentState) -> Dict[str, Any]:
        """

        合成器节点：综合所有研究结果以生成最终答案。
        注意：在引入写作者节点后，此节点可能变得多余，或在写作者失败时作为后备。
        """
        print("Entering 'synthesizer' node: Synthesizer is synthesizing the answer...")
        formatted_results = []
        for res in state["research_results"]:
            if isinstance(res, SearchResult):
                formatted_results.append(
                    f"标题：{res.title}\n网址：{res.url}\n摘要：{res.snippet}\n---"
                )
            elif isinstance(res, RagResult):
                formatted_results.append(
                    f"内容：{res.content}\n来源：{res.source}\n---"
                )
            elif isinstance(res, str):
                formatted_results.append(res)
            else:
                print(f"警告：研究结果格式意外：{res}")
                formatted_results.append(str(res))

        response = llm.invoke(
            SYNTHESIS_PROMPT.format_messages(
                input=state["input"],
                research_results="\n\n".join(formatted_results)
            )
        )
        final_answer = response.content
        print(f"Synthesis completed.")
        return {
            "final_answer": final_answer,
            "intermediate_steps": state["intermediate_steps"] + [AIMessage(content=f"合成器完成答案生成。")],
            "supervisor_decision": "FINISH"
        }

    def route_supervisor_action(self, state: AgentState) -> str:
        """
        主管路由函数：根据主管的输出决定下一个图节点。
        """
        supervisor_decision = state.get("supervisor_decision", "").strip("'\"")
        print(f"Routing function: Received decision '{supervisor_decision}'")

        if state["consecutive_no_progress_count"] >= NO_PROGRESS_THRESHOLD:
            print(f"路由函数：警告：连续 {state['consecutive_no_progress_count']} 步无进展，强制路由到FAIL。")
            return "FAIL"

        # 只有当 replan_needed 为 True 时，才强制跳转到 OUTLINE_PLANNER
        if state.get("replan_needed"):
            print("路由函数：检测到 replan_needed 标志，强制路由到OUTLINE_PLANNER。")
            # 在这里重置标志，避免无限循环
            state["replan_needed"] = False
            return "OUTLINE_PLANNER"

        if not state["research_results"] and state["step_count"] >= MAX_STEPS - 1:
            print("路由函数：警告：接近最大步骤限制且无研究结果，强制路由到FAIL。")
            return "FAIL"

        if state["final_answer"] and supervisor_decision == "FINISH":
            print("路由函数：检测到最终答案且决策为FINISH，任务结束。")
            return "FINISH"

        if supervisor_decision in ["OUTLINE_PLANNER", "WRITER_PLANNER", "EXECUTOR", "EVALUATOR", "WRITER", "REVIEWER", "SYNTHESIZER", "FINISH", "FAIL"]:
            return supervisor_decision
        else:
            print(f"路由函数：主管决策 '{supervisor_decision}' 不明确，默认为OUTLINE_PLANNER。")
            return "OUTLINE_PLANNER"

    def get_app(self):
        """
        返回已编译的LangGraph应用程序。
        """
        return self.app


if __name__ == "__main__":
    # 此块用于测试DeepSearchGraph
    print("Testing deepsearch_graph.py...")

    deep_search_graph = DeepSearchGraph()
    app = deep_search_graph.get_app()

    test_query = "What are the important advances and applications in AI in 2024?"

    print(f"\nStarting DeepSearch agent, query: '{test_query}'")
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
        "consecutive_no_progress_count": 0,
        "planning_attempts_count": 0
    }, {"recursion_limit": 100}):
        if "__end__" not in s:
            print(s)
        else:
            print("\n--- Flow Ended ---")
            final_state = s["__end__"]
            print(f"Final Answer: {final_state.get('final_answer', 'No final answer')}")
            print(f"Research Results: {final_state.get('research_results', 'No research results')}")
            print(f"Chat History: {final_state.get('chat_history', 'No history')}")
            print(f"Final Intermediate Steps: {final_state.get('intermediate_steps', 'No intermediate steps')}")
