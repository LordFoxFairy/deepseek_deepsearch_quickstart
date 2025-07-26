from typing import Dict, Any

from langgraph.graph import StateGraph, END

from backend.src.config.logging_config import get_logger
from backend.src.graphs.search_rag_graph import SearchRagGraph
from backend.src.graphs.writing_graph import WritingGraph
from backend.src.llms.openai_llm import get_chat_model
from backend.src.prompts.agent_decision_prompts import SUPERVISOR_DECISION_PROMPT
from backend.src.schemas.graph_state import AgentState

# --- 全局组件与常量定义 ---
llm = get_chat_model()
MAX_STEPS = 15
NO_PROGRESS_THRESHOLD = 3
logger = get_logger(__name__)


class DeepSearchGraph:
    """
    DeepSearch 主图（Master Graph）。

    该类扮演着一个高级调度器（Orchestrator）的角色，其核心职责是根据子图返回的
    具体状态和决策建议，智能地协调研究子图和写作子图。
    """

    def __init__(self):
        """
        初始化主图实例。
        """
        self.research_graph_app = SearchRagGraph().get_graph()
        self.writing_graph_app = WritingGraph().get_graph()

        self.workflow = StateGraph(AgentState)
        self._add_nodes()
        self._add_edges()
        self.app = self.workflow.compile()

    def _add_nodes(self):
        """
        向主图注册节点。
        """
        self.workflow.add_node("supervisor", self.call_supervisor)
        self.workflow.add_node("research_flow", self.research_graph_app)
        self.workflow.add_node("writing_flow", self.writing_graph_app)

    def _add_edges(self):
        """
        定义主图中节点之间的连接，完全符合Mermaid主图流程。
        """
        self.workflow.set_entry_point("supervisor")
        self.workflow.add_conditional_edges(
            "supervisor",
            self.route_supervisor_action,
            {
                "RESEARCH": "research_flow",
                "WRITING": "writing_flow",
                "FINISH": END,
                "FAIL": END,
            },
        )
        # 任何子图执行完毕后，都将控制权（反馈）交还给 supervisor
        self.workflow.add_edge("research_flow", "supervisor")
        self.workflow.add_edge("writing_flow", "supervisor")

    def call_supervisor(self, state: AgentState) -> Dict[str, Any]:
        """
        主管（Supervisor）节点的核心实现。
        """
        logger.info(f"\n==================== [主图 步骤 {state['step_count']}] ====================")
        logger.info("进入 'supervisor' 节点：主管正在决策...")

        new_step_count = state['step_count'] + 1
        logger.info(f"当前步数更新为: {new_step_count}")

        if new_step_count > MAX_STEPS:
            logger.warning(f"已达到最大步骤数限制 ({MAX_STEPS})，强制终止。")
            return {
                "supervisor_decision": "FAIL",
                "final_answer": "任务因达到最大步骤限制而终止，未能完成。",
                "step_count": new_step_count
            }
        if state["consecutive_no_progress_count"] >= NO_PROGRESS_THRESHOLD:
            logger.warning(f"连续 {state['consecutive_no_progress_count']} 步无进展，强制失败。")
            return {
                "supervisor_decision": "FAIL",
                "final_answer": "任务因连续无进展而终止，未能完成。",
                "step_count": new_step_count
            }

        # 优先采纳子图的明确反馈，这是实现“写作->搜索”的关键
        subgraph_decision = state.get("supervisor_decision")
        if subgraph_decision in ["RESEARCH", "WRITING", "FINISH", "FAIL"]:
            logger.info(f"主管：接收到来自子图的明确指令 '{subgraph_decision}'，将直接执行。")
            return {"supervisor_decision": subgraph_decision, "step_count": new_step_count}

        # 如果没有明确指令（例如，任务刚开始），则调用LLM进行高层决策
        logger.info("主管：未收到子图指令，调用LLM进行高层决策。")
        messages_for_supervisor = state["chat_history"] + state["intermediate_steps"]
        response = llm.invoke(
            SUPERVISOR_DECISION_PROMPT.format_messages(
                current_state=str(state),
                input=state["input"],
                intermediate_steps=messages_for_supervisor
            )
        )

        raw_decision = response.content.strip().upper()
        logger.info(f"主管原始响应: '{raw_decision}'")

        # 解析LLM的宏观决策
        if "PLANNER" in raw_decision or "EXECUTOR" in raw_decision or "EVALUATOR" in raw_decision:
            supervisor_decision = "RESEARCH"
        elif "WRITER" in raw_decision or "REVIEWER" in raw_decision or "SYNTHESIZER" in raw_decision:
            supervisor_decision = "WRITING"
        elif "FINISH" in raw_decision:
            supervisor_decision = "FINISH"
        else:
            if new_step_count <= 1:
                logger.info("主管：检测到初始步骤（冷启动），强制启动研究流程。")
                supervisor_decision = "RESEARCH"
            else:
                supervisor_decision = "FAIL"

        logger.info(f"解析后的主管决策: {supervisor_decision}")

        return {"supervisor_decision": supervisor_decision, "step_count": new_step_count}

    def route_supervisor_action(self, state: AgentState) -> str:
        """
        主管路由函数。
        """
        supervisor_decision = state.get("supervisor_decision", "FAIL")
        logger.info(f"路由函数：接收到决策 '{supervisor_decision}'，将路由到 '{supervisor_decision}' 流程。")
        return supervisor_decision

    def get_app(self):
        """
        提供对已编译的LangGraph可执行应用的公共访问接口。
        """
        return self.app
