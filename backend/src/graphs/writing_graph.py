from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import AgentExecutor, create_tool_calling_agent
from backend.src.tools.rag_tools import rag_tool
from backend.src.config.logging_config import get_logger
from backend.src.llms.openai_llm import get_chat_model
from backend.src.prompts.planner_prompts import WRITER_PLANNER_PROMPT
from backend.src.prompts.writer_prompts import WRITER_PROMPT
from backend.src.prompts.writer_reviewer_prompts import WRITER_REVIEWER_PROMPT
from backend.src.schemas.graph_state import AgentState

llm = get_chat_model()
logger = get_logger(__name__)


class WritingGraph:
    """
    写作思考子图 (Writing Thinking Subgraph)。

    它封装了从“规划写作大纲”到“撰写报告”再到“内部评审与修订”的完整自治循环。
    只有在评审后认为报告合格，或发现需要补充研究时，才会结束并向主图反馈。
    """

    def __init__(self):
        # --- 初始化 Agent 执行器 ---
        # 1. 定义 Agent 可以使用的工具列表。根据你的设计，这里只有 rag_tool。
        tools = [rag_tool]
        # 2. 使用 LangChain 的 `create_tool_calling_agent` 函数创建一个 Agent。
        #    它将 LLM、工具列表和我们精心设计的 WRITER_PROMPT 绑定在一起。
        writing_agent = create_tool_calling_agent(llm, tools, WRITER_PROMPT)
        # 3. 将创建好的 Agent 包装成一个 AgentExecutor。
        #    Executor 负责处理 Agent 的完整运行逻辑，包括调用 Agent、解析工具、执行工具、返回结果等。
        self.agent_executor = AgentExecutor(
            agent=writing_agent,
            tools=tools,
            verbose=True,  # 设置为 True 可以在控制台看到 Agent 的详细思考过程
            handle_parsing_errors=True,  # 帮助处理一些潜在的解析错误
        )

        self.graph = self._build_graph()

    def call_writer_planner(self, state: AgentState) -> dict:
        """
        写作规划器节点。(此节点保持不变)
        """
        logger.info("--- [进入 写作子图: writer_planner 节点] ---")
        response = llm.invoke(
            WRITER_PLANNER_PROMPT.format_messages(
                query=state["input"],
                search_results=state["research_results"]
            )
        )
        writing_plan_steps = [step.strip() for step in response.content.split('\n') if step.strip()]
        logger.info(f"生成的写作大纲: {writing_plan_steps}")
        logger.info("--- [退出 写作子图: writer_planner 节点] ---")
        return {
            "plan": writing_plan_steps,
            "intermediate_steps": state["intermediate_steps"] + [
                AIMessage(content=f"写作规划器完成规划。计划：{writing_plan_steps}")],
        }

    def call_writer(self, state: AgentState) -> dict:
        """
        作者节点 (Agentic 模式)。
        """
        logger.info("--- [进入 写作子图: writer 节点 (Agentic)] ---")

        agent_inputs = {
            "input": state["input"],
            "plan": state["plan"],
        }

        # 调用 Agent 执行器。它会根据 WRITER_PROMPT 的指令，
        # 自动决定何时以及如何调用 rag_tool 来获取信息，以完成写作任务。
        agent_response = self.agent_executor.invoke(agent_inputs)

        final_answer = agent_response["output"]
        writer_steps = agent_response.get("intermediate_steps", [])

        logger.info("写作者 Agent 完成报告起草。")
        logger.info("--- [退出 写作子图: writer 节点] ---")

        # 4. 将 Agent 的思考步骤返回，以便 LangGraph 将其添加到全局的 intermediate_steps 列表中。
        return {
            "final_answer": final_answer,
            "intermediate_steps": writer_steps
        }
    def call_reviewer(self, state: AgentState) -> dict:
        """
        审阅者节点：对报告进行质量评估。(此节点保持不变)
        """
        logger.info("--- [进入 写作子图: reviewer 节点] ---")
        response = llm.invoke(
            WRITER_REVIEWER_PROMPT.format_messages(
                input=state["input"],
                research_results=state["research_results"],
                generated_answer=state["final_answer"]
            )
        )
        review_result = response.content.strip().upper()
        logger.info(f"审阅结果: {review_result}")

        suggested_supervisor_decision = "FAIL"
        if "FINISH" in review_result:
            logger.info("审阅者：报告质量合格，建议完成任务。")
            suggested_supervisor_decision = "FINISH"
        elif "WRITER_PLANNER" in review_result:
            logger.info("审阅者：报告结构或内容需改进，建议重新规划写作。")
            suggested_supervisor_decision = "WRITING"
        elif "OUTLINE_PLANNER" in review_result:
            logger.info("审阅者：研究信息不足，建议返回主图进行补充研究。")
            suggested_supervisor_decision = "RESEARCH"

        logger.info("--- [退出 写作子图: reviewer 节点] ---")
        return {
            "evaluation_results": review_result,
            "intermediate_steps": state["intermediate_steps"] + [
                AIMessage(content=f"审阅者完成审阅。建议：{suggested_supervisor_decision}")],
            "supervisor_decision": suggested_supervisor_decision
        }

    def route_writing_decision(self, state: AgentState) -> str:
        """
        写作子图内部的条件路由。(此节点保持不变)
        """
        logger.info("--- [写作子图 路由] ---")
        decision = state.get("supervisor_decision", "FAIL")

        if decision == "WRITING":
            logger.info("决策：修订写作大纲（内部循环）。")
            return "revise_plan"
        else:
            logger.info(f"决策：结束写作子图 (向主图反馈: {decision})。")
            return "end_writing"

    def _build_graph(self) -> CompiledStateGraph:
        """
        构建包含“规划-写作-评审-修订”循环的自治写作子图。(此节点保持不变)
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("writer_planner", self.call_writer_planner)
        workflow.add_node("writer", self.call_writer)
        workflow.add_node("reviewer", self.call_reviewer)

        workflow.set_entry_point("writer_planner")

        workflow.add_edge("writer_planner", "writer")
        workflow.add_edge("writer", "reviewer")

        workflow.add_conditional_edges(
            "reviewer",
            self.route_writing_decision,
            {
                "end_writing": END,
                "revise_plan": "writer_planner",
            },
        )

        return workflow.compile()

    def get_graph(self) -> CompiledStateGraph:
        return self.graph
