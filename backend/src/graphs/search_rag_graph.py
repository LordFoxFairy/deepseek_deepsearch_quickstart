import json
import time
import re
from typing import Dict, Any, List, Optional, Tuple

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from backend.src.config.logging_config import get_logger
from backend.src.llms.openai_llm import get_chat_model
from backend.src.prompts.evaluator_prompts import EVALUATOR_PROMPT
from backend.src.prompts.planner_prompts import OUTLINE_PLANNER_PROMPT
from backend.src.prompts.summarizer_prompts import RESEARCH_SUMMARIZER_PROMPT
from backend.src.schemas.graph_state import AgentState, PlanItem
from backend.src.services.llama_index_service import llama_index_service  # 确保导入
from backend.src.tools.search_tools import search_tool

llm = get_chat_model()
logger = get_logger(__name__)


def _clean_json_from_llm(llm_output: str) -> str:
    match = re.search(r"```(?:json)?(.*)```", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return llm_output.strip()


class SearchRagGraph:
    def __init__(self):
        self.graph = self._build_graph()

    def call_outline_planner(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- [进入 研究子图: outline_planner 节点] ---")
        state_summary = {
            "input": state.get("input"),
            "research_plan_status": [f"'{item['item_id']}': {item['status']}" for item in
                                     state.get("research_plan", [])],
            "error_log_count": len(state.get("error_log", []))
        }
        try:
            planner_chain = OUTLINE_PLANNER_PROMPT | llm
            response = planner_chain.invoke({
                "query": state["input"],
                "current_state": json.dumps(state_summary, ensure_ascii=False, indent=2),
                "failure_context": "", "no_progress_context": ""
            })
            cleaned_json_string = _clean_json_from_llm(response.content)
            plan_items_json = json.loads(cleaned_json_string)
            research_plan: List[PlanItem] = []
            for item_json in plan_items_json:
                plan_item: PlanItem = {
                    "item_id": item_json.get("item_id"), "description": item_json.get("description"),
                    "dependencies": item_json.get("dependencies", []), "status": "pending", "content": "",
                    "summary": None, "execution_log": [], "evaluation_results": None, "attempt_count": 0
                }
                research_plan.append(plan_item)
            logger.info(f"成功生成结构化研究计划，共 {len(research_plan)} 项。")
            return {"research_plan": research_plan}
        except Exception as e:
            logger.error(f"在 outline_planner 中发生未知错误: {e}", exc_info=True)
            logger.error(f"LLM原始输出: {response.content if 'response' in locals() else 'N/A'}")
            return {"error_log": [{"node": "outline_planner", "error": str(e)}]}

    def call_executor(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- [进入 研究子图: executor 节点] ---")
        current_item_id: Optional[str] = state.get("current_plan_item_id")
        if not current_item_id: return {"error_log": [{"node": "executor", "error": "Missing current_plan_item_id"}]}
        plan: List[PlanItem] = state.get("research_plan", [])
        item_to_execute, item_index = self._find_plan_item(plan, current_item_id)
        if not item_to_execute: return {
            "error_log": [{"node": "executor", "error": f"PlanItem with id {current_item_id} not found"}]}
        item_to_execute["status"] = "in_progress"
        item_to_execute["attempt_count"] += 1
        query = item_to_execute["description"]
        logger.info(f"第 {item_to_execute['attempt_count']} 次尝试执行任务 '{current_item_id}': 搜索 '{query}'")
        try:
            # 1. 调用搜索工具
            search_results_list = search_tool.invoke({"query": query})
            time.sleep(2)

            # 2. 准备数据用于存储和返回
            serializable_output = [res.model_dump() for res in search_results_list]
            item_to_execute["content"] = json.dumps(serializable_output, ensure_ascii=False, indent=2)
            item_to_execute["execution_log"].append(f"成功执行搜索，获得 {len(search_results_list)} 条结果。")

            # 3. 将搜索结果存入 LlamaIndex 知识库
            llama_index_service.add_search_results_to_index(search_results_list)

            logger.info("搜索工具执行完毕，并已将结果存入知识库。")

        except Exception as e:
            error_message = f"工具 'search_tool' 执行失败: {e}"
            logger.error(error_message, exc_info=True)
            item_to_execute["status"] = "failed"
            item_to_execute["execution_log"].append(error_message)
            state["error_log"].append({"node": "executor", "item_id": current_item_id, "error": str(e)})

        plan[item_index] = item_to_execute
        return {"research_plan": plan, "research_results": state.get("research_results", []) + serializable_output}

    def call_summarizer(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- [进入 研究子图: summarizer 节点] ---")
        current_item_id: Optional[str] = state.get("current_plan_item_id")
        if not current_item_id: return {"error_log": [{"node": "summarizer", "error": "Missing current_plan_item_id"}]}
        plan: List[PlanItem] = state.get("research_plan", [])
        item_to_summarize, item_index = self._find_plan_item(plan, current_item_id)
        if not item_to_summarize or item_to_summarize.get("status") == "failed": return {}
        raw_results = json.loads(item_to_summarize.get("content", "[]"))
        snippets = [res.get("snippet", "") for res in raw_results]
        if not snippets:
            logger.warning(f"摘要器：任务 '{current_item_id}' 没有可供摘要的内容。")
            item_to_summarize["content"] = "无有效信息可供摘要。"
            plan[item_index] = item_to_summarize
            return {"research_plan": plan}
        try:
            summarizer_chain = RESEARCH_SUMMARIZER_PROMPT | llm
            response = summarizer_chain.invoke({
                "topic": item_to_summarize["description"],
                "search_results_content": "\n---\n".join(snippets)
            })
            summary = response.content.strip()
            item_to_summarize["content"] = summary
            item_to_summarize["execution_log"].append("已生成研究结果的高质量摘要。")
            logger.info(f"摘要器：已为任务 '{current_item_id}' 生成摘要。")
        except Exception as e:
            error_message = f"摘要器执行失败: {e}"
            logger.error(error_message, exc_info=True)
            item_to_summarize["execution_log"].append(error_message)
            state["error_log"].append({"node": "summarizer", "item_id": current_item_id, "error": str(e)})
        plan[item_index] = item_to_summarize
        return {"research_plan": plan}

    def call_evaluator(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- [进入 研究子图: evaluator 节点] ---")
        current_item_id: Optional[str] = state.get("current_plan_item_id")
        if not current_item_id: return {"error_log": [{"node": "evaluator", "error": "Missing current_plan_item_id"}]}
        plan: List[PlanItem] = state.get("research_plan", [])
        item_to_evaluate, item_index = self._find_plan_item(plan, current_item_id)
        if not item_to_evaluate or item_to_evaluate.get("status") == "failed": return {}
        try:
            evaluator_chain = EVALUATOR_PROMPT | llm
            response = evaluator_chain.invoke({
                "input": state["input"],
                "plan_item_description": item_to_evaluate["description"],
                "plan_item_content": item_to_evaluate["content"]
            })
            cleaned_json_string = _clean_json_from_llm(response.content)
            eval_result = json.loads(cleaned_json_string)
            item_to_evaluate["evaluation_results"] = json.dumps(eval_result, ensure_ascii=False)
            if eval_result.get("is_sufficient"):
                item_to_evaluate["status"] = "completed"
                logger.info(f"评估器：任务 '{current_item_id}' 已完成。")
            else:
                item_to_evaluate["status"] = "completed"
                logger.warning(f"评估器：任务 '{current_item_id}' 已完成，但结果被认为不充分。")
        except Exception as e:
            error_message = f"评估器执行失败: {e}"
            logger.error(error_message, exc_info=True)
            logger.error(f"LLM原始输出: {response.content if 'response' in locals() else 'N/A'}")
            item_to_evaluate["status"] = "failed"
            item_to_evaluate["evaluation_results"] = "评估器输出格式错误。"
            state["error_log"].append({"node": "evaluator", "item_id": current_item_id, "error": str(e)})
        plan[item_index] = item_to_evaluate
        return {"research_plan": plan}

    def _find_plan_item(self, plan: List[PlanItem], item_id: str) -> Tuple[Optional[PlanItem], int]:
        for i, item in enumerate(plan):
            if item["item_id"] == item_id: return item, i
        return None, -1

    def _route_entry(self, state: AgentState) -> str:
        if not state.get("research_plan"): return "outline_planner"
        if not state.get("current_plan_item_id"): return END
        return "executor"

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("outline_planner", self.call_outline_planner)
        workflow.add_node("executor", self.call_executor)
        workflow.add_node("summarizer", self.call_summarizer)
        workflow.add_node("evaluator", self.call_evaluator)
        workflow.set_conditional_entry_point(self._route_entry)
        workflow.add_edge("outline_planner", END)
        workflow.add_edge("executor", "summarizer")
        workflow.add_edge("summarizer", "evaluator")
        workflow.add_edge("evaluator", END)
        return workflow.compile()

    def get_graph(self) -> CompiledStateGraph:
        return self.graph
