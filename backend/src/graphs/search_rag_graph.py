import ast
import time

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from backend.src.config.logging_config import get_logger
from backend.src.llms.openai_llm import get_chat_model
from backend.src.prompts.evaluator_prompts import EVALUATOR_PROMPT
from backend.src.prompts.planner_prompts import OUTLINE_PLANNER_PROMPT
from backend.src.schemas.graph_state import AgentState
from backend.src.services.llama_index_service import llama_index_service
from backend.src.tools.search_tools import search_tool

# --- 全局组件与常量定义 ---
llm = get_chat_model()
PLANNING_ATTEMPTS_THRESHOLD = 3
logger = get_logger(__name__)


class SearchRagGraph:
    """
    搜索思考子图 (Search Thinking Subgraph)。

    该子图作为一个独立的研究单元，负责从规划研究、执行搜索到评估结果的完整信息收集流程。
    它具备内部循环能力，可以根据评估结果自我修正研究计划，直到产出足够高质量的信息。
    """

    def __init__(self):
        self.graph = self._build_graph()

    def call_outline_planner(self, state: AgentState) -> dict:
        """
        规划器节点：生成研究计划。
        """
        logger.info("--- [进入 搜索子图: outline_planner 节点] ---")
        state["planning_attempts_count"] += 1
        logger.info(f"当前规划尝试次数: {state['planning_attempts_count']}")

        failure_context = ""
        if state.get("replan_needed"):
            failure_context += "注意：先前的评估器建议重新规划。请分析原因并调整计划。\n"
        no_progress_context = ""
        if state["consecutive_no_progress_count"] > 0:
            no_progress_context = f"注意：代理已连续 {state['consecutive_no_progress_count']} 步未能取得有效的研究进展。\n"

        response = llm.invoke(
            OUTLINE_PLANNER_PROMPT.format_messages(
                query=state["input"],
                current_state=state,
                failure_context=failure_context or "没有特定的失败上下文。",
                no_progress_context=no_progress_context or "没有连续无进展的上下文。"
            )
        )

        plan_steps = []
        try:
            plan_steps = ast.literal_eval(response.content.strip())
        except (ValueError, SyntaxError):
            logger.warning(f"LLM未能按要求输出合法的Python列表。原始输出: {response.content}")
            plan_steps = [response.content.strip()]

        logger.info(f"格式化后的研究计划: {plan_steps}")
        logger.info("--- [退出 搜索子图: outline_planner 节点] ---")
        return {"plan": plan_steps, "supervisor_decision": "EXECUTOR",
                "planning_attempts_count": state["planning_attempts_count"]}

    def call_executor(self, state: AgentState) -> dict:
        """
        执行器节点：遍历整个计划并对每一步执行搜索。
        """
        logger.info("--- [进入 搜索子图: executor 节点] ---")
        plan = state.get("plan", [])
        if not plan:
            logger.error("错误：没有计划可执行。")
            return {"tool_output": "错误：没有计划可执行。", "replan_needed": True, "supervisor_decision": "EVALUATOR"}

        tool_output_list = []
        initial_research_results_count = len(state["research_results"])

        for step in plan:
            if not isinstance(step, str) or not step.strip():
                continue

            try:
                logger.info(f"执行器：调用 search_tool，查询: '{step}'")
                output = search_tool.invoke({"query": step})
                tool_output_list.append(f"查询 '{step}' 的结果: {len(output)} 条")
                logger.info(f"工具 'search_tool' 已成功执行，查询: '{step}'。")

                state["research_results"].extend(output)
                # for res in output:
                #     if isinstance(res, dict) and 'snippet' in res:
                #         state["raw_research_content"] += res['snippet'] + "\n"

                llama_index_service.add_search_results_to_index(output)

                # 在每次API调用后，暂停2秒。
                # 这可以极大地降低因为请求过于频繁而被搜索引擎（如Google）暂时屏蔽的风险（429错误）。
                logger.info("为避免请求过于频繁，暂停2秒...")
                time.sleep(2)



            except Exception as e:
                error_message = f"工具 'search_tool' 在查询 '{step}' 时执行失败：{e}"
                tool_output_list.append(error_message)
                logger.error(error_message)

        logger.info("LlamaIndex 数据添加完成。")

        if len(state["research_results"]) == initial_research_results_count:
            state["consecutive_no_progress_count"] += 1
            logger.warning(f"执行器未取得新的研究进展，连续无进展计数：{state['consecutive_no_progress_count']}")
        else:
            state["consecutive_no_progress_count"] = 0
            logger.info("执行器取得了新的研究进展，重置连续无进展计数。")

        executor_log_content = f"执行器完成了对 {len(plan)} 个步骤的搜索。"
        logger.info("--- [退出 搜索子图: executor 节点] ---")
        return {
            "tool_output": "\n".join(tool_output_list),
            "intermediate_steps": state["intermediate_steps"] + [AIMessage(content=executor_log_content)],
            "consecutive_no_progress_count": state["consecutive_no_progress_count"],
            "supervisor_decision": "EVALUATOR"
        }

    def call_evaluator(self, state: AgentState) -> dict:
        """
        评估器节点：评估研究结果的质量和完整性。
        """
        logger.info("--- [进入 搜索子图: evaluator 节点] ---")
        evaluation_context = ""
        has_research_results = bool(state["research_results"])
        if not has_research_results and state["step_count"] > 1:
            evaluation_context += "注意：尚未收集到有效的研究结果。\n"
        no_progress_context = ""
        if state["consecutive_no_progress_count"] >= 3:
            no_progress_context = f"警告：代理已连续 {state['consecutive_no_progress_count']} 步未能取得有效的研究进展。\n"

        response = llm.invoke(
            EVALUATOR_PROMPT.format_messages(
                input=state["input"],
                current_state=state,
                research_results=state["research_results"],
                final_answer=state.get("final_answer"),
                evaluation_context=evaluation_context,
                no_progress_context=no_progress_context
            )
        )
        evaluation_result = response.content.strip()
        logger.info(f"评估结果: {evaluation_result}")

        llm_decision = evaluation_result.upper()
        replan_needed = False

        if "FAIL" in llm_decision or state["consecutive_no_progress_count"] >= 3:
            logger.warning("评估器：检测到无法修复的错误或持续无进展，建议任务失败。")
            suggested_supervisor_decision = "FAIL"
        elif "WRITER" in llm_decision:
            logger.info("评估器：研究结果充足，建议进入写作流程。")
            suggested_supervisor_decision = "WRITING"
        elif has_research_results and state["planning_attempts_count"] >= PLANNING_ATTEMPTS_THRESHOLD:
            logger.warning(
                f"评估器：虽有结果，但已达最大研究尝试次数({PLANNING_ATTEMPTS_THRESHOLD})，强制建议进入写作流程。")
            suggested_supervisor_decision = "WRITING"
        else:
            logger.info("评估器：研究结果不足或需调整策略，建议在子图内部重新规划研究。")
            replan_needed = True
            suggested_supervisor_decision = "RESEARCH"

        logger.info("--- [退出 搜索子图: evaluator 节点] ---")
        return {
            "evaluation_results": evaluation_result,
            "replan_needed": replan_needed,
            "intermediate_steps": state["intermediate_steps"] + [
                AIMessage(content=f"评估器完成评估。建议：{suggested_supervisor_decision}")],
            "supervisor_decision": suggested_supervisor_decision
        }

    def route_search_decision(self, state: AgentState) -> str:
        """
        研究子图内部的条件路由，完全符合Mermaid子图流程。
        """
        logger.info("--- [研究子图 路由] ---")
        decision = state.get("supervisor_decision", "FAIL")

        if decision == "RESEARCH":
            logger.info("决策：修订研究计划（内部循环）。")
            return "revise_plan"
        else:
            logger.info(f"决策：结束研究子图 (向主图反馈: {decision})。")
            return "end_search"

    def _build_graph(self) -> CompiledStateGraph:
        """
        构建包含“规划-执行-评估-修订”循环的自治研究子图。
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("outline_planner", self.call_outline_planner)
        workflow.add_node("executor", self.call_executor)
        workflow.add_node("evaluator", self.call_evaluator)

        workflow.set_entry_point("outline_planner")

        workflow.add_edge("outline_planner", "executor")
        workflow.add_edge("executor", "evaluator")

        workflow.add_conditional_edges(
            "evaluator",
            self.route_search_decision,
            {
                "end_search": END,
                "revise_plan": "outline_planner",
            },
        )
        return workflow.compile()

    def get_graph(self) -> CompiledStateGraph:
        return self.graph
