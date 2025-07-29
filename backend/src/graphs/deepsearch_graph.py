import json
import re
from typing import Dict, Any, List, Tuple

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from backend.src.config.logging_config import get_logger
from backend.src.llms.openai_llm import get_chat_model
from backend.src.schemas.graph_state import AgentState, PlanItem
from backend.src.graphs.research_executor import execute_research_task
from backend.src.graphs.writing_executor import execute_writing_task, final_assembler
from backend.src.prompts.planner_prompts import MASTER_PLANNER_PROMPT
from backend.src.prompts.summarizer_prompts import RESEARCH_SUMMARIZER_PROMPT, OVERALL_REPORT_SUMMARIZER_PROMPT

llm = get_chat_model()
logger = get_logger(__name__)


def _clean_json_from_llm(llm_output: str) -> str:
    """从LLM的输出中提取并清理JSON字符串。"""
    match = re.search(r"```(?:json)?(.*)```", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return llm_output.strip()


class DeepSearchGraph:
    """
    报告生成主图
    """

    def __init__(self):
        self.workflow = self._build_graph()

    def get_app(self):
        return self.workflow

    def _validate_and_correct_plan(self, plan: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """计划审查员：验证并修正计划的依赖关系。"""
        logger.info("--- [计划审查] 开始审查计划依赖关系... ---")
        errors = []
        corrected_plan = list(plan)
        research_task_ids = {item['item_id'] for item in corrected_plan if item['task_type'] == 'RESEARCH'}
        all_task_ids = {item['item_id'] for item in corrected_plan}

        for i, item in enumerate(corrected_plan):
            if item['task_type'] == 'WRITING':
                original_deps = item.get('dependencies', [])
                corrected_deps = []
                for dep_id in original_deps:
                    if dep_id not in all_task_ids:
                        errors.append(f"依赖 '{dep_id}' 不存在，已移除。")
                        continue
                    if dep_id not in research_task_ids:
                        errors.append(f"写作任务 '{item['item_id']}' 错误地依赖了非研究任务 '{dep_id}'，已移除。")
                        continue
                    corrected_deps.append(dep_id)
                corrected_plan[i]['dependencies'] = corrected_deps

        if errors:
            logger.warning(f"计划审查发现并修正了 {len(errors)} 个错误: {errors}")
        else:
            logger.info("--- [计划审查] 审查完成，未发现依赖错误。 ---")
        return corrected_plan, errors

    async def call_planner(self, state: AgentState) -> Dict[str, Any]:
        """节点 1: 生成初始计划，并由“审查员”进行校验。"""
        logger.info("--- [阶段 1] 进入 planner 节点 ---")
        prompt = MASTER_PLANNER_PROMPT.format(query=state['input'])
        response = await llm.ainvoke(prompt)
        cleaned_json = _clean_json_from_llm(response.content)
        try:
            plan_data = json.loads(cleaned_json)
            raw_plan_list = plan_data.get("plan", [])
            overall_outline = plan_data.get("overall_outline", "")

            plan_items = []
            for item in raw_plan_list:
                if isinstance(item, dict):
                    complete_item_data = {
                        "item_id": item.get("item_id"), "task_type": item.get("task_type"),
                        "description": item.get("description"), "dependencies": item.get("dependencies", []),
                        "status": "pending", "content": "", "summary": None, "execution_log": [],
                        "evaluation_results": None, "attempt_count": 0,
                    }
                    plan_items.append(complete_item_data)

            corrected_plan, validation_errors = self._validate_and_correct_plan(plan_items)
            final_plan = [PlanItem(**item) for item in corrected_plan]
            error_log = state.get("error_log", [])
            if validation_errors:
                error_log.append({"node": "planner_validator", "errors": validation_errors})

            logger.info(f"成功生成并审查了 {len(final_plan)} 个计划项。")
            return {"plan": final_plan, "overall_outline": overall_outline, "error_log": error_log}
        except json.JSONDecodeError:
            logger.error(f"Planner 输出的 JSON 格式无效: {cleaned_json}")
            return {"plan": [], "overall_outline": "", "error_log": [{"node": "planner", "error": "无法解析计划"}]}

    # --- 研究阶段 supervisor/executor 模式 ---
    def research_supervisor(self, state: AgentState) -> Dict[str, Any]:
        """节点 2: 研究主管 - 决定下一个研究任务。"""
        logger.info("--- [阶段 2] 进入 research_supervisor 节点 ---")
        plan = state.get("plan", [])
        next_research_task = next(
            (task for task in plan if task.get("task_type") == "RESEARCH" and task.get("status") != "completed"), None)
        if next_research_task:
            logger.info(f"研究主管决策：委派下一个任务 '{next_research_task['description']}'")
            return {"current_plan_item_id": next_research_task['item_id']}
        else:
            logger.info("研究主管决策：所有研究任务已完成，进入计划摘要阶段。")
            return {"current_plan_item_id": None}

    def route_research_action(self, state: AgentState) -> str:
        """根据研究主管的决策进行路由。"""
        return "research_executor" if state.get("current_plan_item_id") else "plan_summarizer"

    async def call_plan_summarizer(self, state: AgentState) -> Dict[str, Any]:
        """节点 3: 为每个写作任务生成其对应的章节摘要。"""
        logger.info("--- [阶段 3] 进入 plan_summarizer 节点 ---")
        plan = state.get("plan", [])
        updated_plan = list(plan)
        for i, item in enumerate(updated_plan):
            if item.get("task_type") == "WRITING":
                dependencies = item.get("dependencies", [])
                research_content = [
                    f"研究任务 '{next((p.get('description') for p in updated_plan if p.get('item_id') == dep_id), '')}':\n{next((p.get('content') for p in updated_plan if p.get('item_id') == dep_id), '')}"
                    for dep_id in dependencies]
                if not any(research_content):
                    updated_plan[i]["content"] = f"本章旨在探讨 '{item.get('description')}'，但未能找到相关的研究资料。"
                    continue
                prompt = RESEARCH_SUMMARIZER_PROMPT.format(topic=item.get("description"),
                                                           search_results_content="\n\n".join(research_content))
                response = await llm.ainvoke(prompt)
                updated_plan[i]["content"] = response.content
        return {"plan": updated_plan}

    async def generate_overall_summary(self, state: AgentState) -> Dict[str, Any]:
        """节点 3.5: 生成整篇文章的核心摘要。"""
        logger.info("--- [阶段 3.5] 进入 generate_overall_summary 节点 ---")
        plan = state.get("plan", [])
        all_writing_tasks = [t for t in plan if t.get("task_type") == "WRITING"]
        if not all_writing_tasks: return {}
        all_chapter_summaries = "\n\n---\n\n".join(
            [f"章节目标: {t.get('description', '无描述')}\n核心内容摘要: {t.get('content', '摘要不可用。')}" for t in
             all_writing_tasks])
        prompt = OVERALL_REPORT_SUMMARIZER_PROMPT.format(all_chapter_summaries=all_chapter_summaries)
        response = await llm.ainvoke(prompt)
        return {"overall_outline": response.content}

    def writing_supervisor(self, state: AgentState) -> Dict[str, Any]:
        """节点 4: 写作主管 - 决定下一个写作任务。"""
        plan = state.get("plan", [])
        next_writing_task = next(
            (task for task in plan if task.get("task_type") == "WRITING" and task.get("status") != "completed"), None)
        return {"current_plan_item_id": next_writing_task['item_id'] if next_writing_task else None}

    def route_writing_action(self, state: AgentState) -> str:
        """根据写作主管的决策进行路由。"""
        return "writing_executor" if state.get("current_plan_item_id") else "final_assembler"

    def _build_graph(self) -> CompiledStateGraph:
        """构建新的、支持逐章研究和写作的工作流。"""
        workflow = StateGraph(AgentState)
        workflow.add_node("planner", self.call_planner)
        workflow.add_node("research_supervisor", self.research_supervisor)
        workflow.add_node("research_executor", execute_research_task)
        workflow.add_node("plan_summarizer", self.call_plan_summarizer)
        workflow.add_node("generate_overall_summary", self.generate_overall_summary)
        workflow.add_node("writing_supervisor", self.writing_supervisor)
        workflow.add_node("writing_executor", execute_writing_task)
        workflow.add_node("final_assembler", final_assembler)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "research_supervisor")

        # 添加研究循环
        workflow.add_conditional_edges("research_supervisor", self.route_research_action,
                                       {"research_executor": "research_executor", "plan_summarizer": "plan_summarizer"})
        workflow.add_edge("research_executor", "research_supervisor")

        workflow.add_edge("plan_summarizer", "generate_overall_summary")
        workflow.add_edge("generate_overall_summary", "writing_supervisor")

        workflow.add_conditional_edges("writing_supervisor", self.route_writing_action,
                                       {"writing_executor": "writing_executor", "final_assembler": "final_assembler"})
        workflow.add_edge("writing_executor", "writing_supervisor")
        workflow.add_edge("final_assembler", END)

        app = workflow.compile()
        logger.info("DeepSearchGraph 构建完成。")
        return app
