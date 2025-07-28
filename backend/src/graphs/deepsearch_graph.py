import json
import re
from typing import Dict, Any, Set

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from backend.src.config.logging_config import get_logger
from backend.src.graphs.search_rag_graph import SearchRagGraph
from backend.src.graphs.writing_graph import WritingGraph
from backend.src.llms.openai_llm import get_chat_model
from backend.src.prompts.agent_decision_prompts import SUPERVISOR_DECISION_PROMPT
from backend.src.schemas.graph_state import AgentState

# --- 全局组件与常量定义 ---
llm = get_chat_model()
MAX_STEPS = 30
logger = get_logger(__name__)


def _clean_json_from_llm(llm_output: str) -> str:
    """健壮的辅助函数，用于清理LLM输出的字符串，从中提取纯净的JSON。"""
    match = re.search(r"```(?:json)?(.*)```", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return llm_output.strip()


class DeepSearchGraph:
    def __init__(self):
        self.research_graph_app = SearchRagGraph().get_graph()
        self.writing_graph_app = WritingGraph().get_graph()
        self.workflow = self._build_graph()

    def _update_plan_item_statuses(self, state: AgentState) -> AgentState:
        completed_ids: Set[str] = set()
        for plan in [state.get("research_plan", []), state.get("writing_plan", [])]:
            if not plan: continue
            for item in plan:
                if item["status"] == "completed":
                    completed_ids.add(item["item_id"])
        for plan_name in ["research_plan", "writing_plan"]:
            plan = state.get(plan_name, [])
            if not plan: continue
            for item in plan:
                if item["status"] == "pending":
                    if all(dep_id in completed_ids for dep_id in item["dependencies"]):
                        item["status"] = "ready"
                        logger.info(f"状态更新：任务 '{item['item_id']}' 的依赖已满足，状态变更为 'ready'。")
        return state

    def call_supervisor(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"\n==================== [主图 步骤 {state.get('step_count', 0)}] ====================")
        logger.info("进入 'supervisor' 节点：总指挥官正在决策...")
        step_count = state.get('step_count', 0) + 1
        if step_count > MAX_STEPS:
            logger.error("任务因达到最大步骤限制而终止。")
            return {"step_count": step_count, "supervisor_decision": "FAIL",
                    "final_answer": "任务因达到最大步骤限制而终止。"}

        state = self._update_plan_item_statuses(state)

        writing_plan = state.get("writing_plan", [])
        if writing_plan and all(item['status'] == 'completed' for item in writing_plan):
            # 如果最终报告已经生成，则任务完成
            if state.get("final_answer"):
                logger.info("总指挥官决策：最终报告已生成，任务完成。")
                return {"step_count": step_count, "supervisor_decision": "FINISH"}
            # 否则，进入整合阶段
            else:
                logger.info("总指挥官决策：所有写作任务已完成，强制进入最终整合阶段。")
                return {"step_count": step_count, "supervisor_decision": "SYNTHESIZE", "current_plan_item_id": None}

        if not state.get("research_plan"):
            logger.info("总指挥官决策：启动研究流程。")
            return {"step_count": step_count, "supervisor_decision": "RESEARCH", "current_plan_item_id": None}

        research_plan = state.get("research_plan", [])
        if research_plan and all(item['status'] == 'completed' for item in research_plan) and not writing_plan:
            logger.info("总指挥官决策：研究完成，启动写作流程。")
            return {"step_count": step_count, "supervisor_decision": "WRITING", "current_plan_item_id": None}

        try:
            supervisor_chain = SUPERVISOR_DECISION_PROMPT | llm
            response = supervisor_chain.invoke({
                "input": state["input"],
                "intermediate_steps": state.get("intermediate_steps", []),
                "research_plan": research_plan,
                "writing_plan": writing_plan,
                "shared_context": state.get("shared_context", {}),
                "error_log": state.get("error_log", [])
            })
            cleaned_json_string = _clean_json_from_llm(response.content)
            decision_json = json.loads(cleaned_json_string)
            next_action = decision_json.get("next_action")
            target_item_id = decision_json.get("target_item_id")
            logger.info(f"总指挥官LLM决策: {next_action}, 目标任务ID: {target_item_id}")
            return {"step_count": step_count, "supervisor_decision": next_action,
                    "current_plan_item_id": target_item_id}
        except Exception as e:
            logger.error(f"总指挥官决策失败，LLM可能返回了非JSON格式: {e}", exc_info=True)
            logger.error(f"LLM原始输出: {response.content if 'response' in locals() else 'N/A'}")
            return {"step_count": step_count, "supervisor_decision": "FAIL"}

    def route_supervisor_action(self, state: AgentState) -> str:
        decision = state.get("supervisor_decision", "FAIL")
        logger.info(f"主图路由：接收到决策 '{decision}'，准备分派任务...")
        return decision

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("supervisor", self.call_supervisor)
        workflow.add_node("research_flow", self.research_graph_app)
        workflow.add_node("writing_flow", self.writing_graph_app)
        workflow.set_entry_point("supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self.route_supervisor_action,
            {
                "RESEARCH": "research_flow", "WRITING": "writing_flow",
                "SYNTHESIZE": "writing_flow", "FINISH": END, "FAIL": END,
            },
        )
        workflow.add_edge("research_flow", "supervisor")
        workflow.add_edge("writing_flow", "supervisor")
        return workflow.compile()

    def get_app(self):
        return self.workflow
