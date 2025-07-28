import json
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from backend.src.config.logging_config import get_logger
from backend.src.llms.openai_llm import get_chat_model
from backend.src.prompts.planner_prompts import WRITER_PLANNER_PROMPT
from backend.src.prompts.summarizer_prompts import CHAPTER_SUMMARIZER_PROMPT
from backend.src.prompts.writer_prompts import WRITER_PROMPT
from backend.src.schemas.graph_state import AgentState, PlanItem
from backend.src.tools.rag_tools import rag_tool

llm = get_chat_model()
logger = get_logger(__name__)


def _find_plan_item(plan: List[PlanItem], item_id: str) -> Tuple[Optional[PlanItem], int]:
    for i, item in enumerate(plan):
        if item["item_id"] == item_id: return item, i
    return None, -1


def _clean_json_from_llm(llm_output: str) -> str:
    match = re.search(r"```(?:json)?(.*)```", llm_output, re.DOTALL)
    if match: return match.group(1).strip() if match.group(1) else match.group(2).strip()
    return llm_output.strip()


async def call_chapter_summarizer(state: AgentState) -> Dict[str, Any]:
    logger.info("--- [进入 写作子图: chapter_summarizer 节点] ---")
    current_item_id = state["current_plan_item_id"]
    writing_plan = state["writing_plan"]
    item, item_index = _find_plan_item(writing_plan, current_item_id)
    if item["status"] != "completed": return {}
    summarizer_chain = CHAPTER_SUMMARIZER_PROMPT | llm
    response = await summarizer_chain.ainvoke({"chapter_content": item["content"]})
    summary = response.content.strip()
    item["summary"] = summary
    item["execution_log"].append("已生成章节摘要。")
    logger.info(f"已为章节 '{item['description']}' 生成摘要。")
    writing_plan[item_index] = item
    return {"writing_plan": writing_plan}


class WritingGraph:
    def __init__(self):
        self.graph = self._build_graph()

    async def call_planner(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- [进入 写作子图: planner 节点] ---")
        research_content = "\n\n".join(
            [f"### {item['description']}\n{item['content']}" for item in state.get("research_plan", [])])
        planner_chain = WRITER_PLANNER_PROMPT | llm
        response = await planner_chain.ainvoke({"query": state["input"], "search_results": research_content})
        cleaned_json_string = _clean_json_from_llm(response.content)
        plan_items_json = json.loads(cleaned_json_string)
        writing_plan: List[PlanItem] = []
        for item_json in plan_items_json:
            plan_item: PlanItem = {
                "item_id": item_json.get("item_id"), "description": item_json.get("description"),
                "dependencies": item_json.get("dependencies", []), "status": "pending", "content": "",
                "summary": None, "execution_log": [], "evaluation_results": None, "attempt_count": 0
            }
            writing_plan.append(plan_item)
        logger.info(f"成功生成结构化写作计划，共 {len(writing_plan)} 项。")
        return {"writing_plan": writing_plan}

    async def call_writer_agent(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- [进入 写作子图: 上下文感知的 writer_agent 节点] ---")
        current_item_id = state["current_plan_item_id"]
        if not current_item_id: raise ValueError("Writer Agent: current_plan_item_id 缺失。")

        writing_plan = state["writing_plan"]
        item, item_index = _find_plan_item(writing_plan, current_item_id)
        if not item: raise ValueError(f"Writer Agent: 未能在写作计划中找到 ID 为 {current_item_id} 的任务。")

        item["status"] = "in_progress"
        item["attempt_count"] += 1
        logger.info(f"第 {item['attempt_count']} 次尝试撰写章节: '{item['description']}' (上下文感知)")

        completed_items = [p for p in writing_plan if p["status"] == "completed"]

        previous_chapter_content = "这是第一章，没有前文。"
        if completed_items:
            previous_chapter_content = completed_items[-1].get("content", "前一章内容为空。")

        other_completed_items = completed_items[:-1] if len(completed_items) > 1 else []
        previous_summaries = "\n".join(
            [f"- {p['description']}: {p['summary']}" for p in other_completed_items if p.get("summary")])
        if not previous_summaries:
            previous_summaries = "无其他章节摘要。"

        revision_notes = item.get("evaluation_results", "无")

        tagged_llm = llm.with_config(tags=["writer_agent"])
        agent = create_openai_tools_agent(tagged_llm, [rag_tool], WRITER_PROMPT)
        agent_executor = AgentExecutor(agent=agent, tools=[rag_tool], verbose=True)

        agent_inputs = {
            "input": state["input"],
            "task_description": item["description"],
            "previous_chapter_content": previous_chapter_content,
            "previous_summaries": previous_summaries,
            "revision_notes": revision_notes,
        }

        raw_content = ""
        async for chunk in agent_executor.astream(agent_inputs):
            if "output" in chunk:
                raw_content = chunk["output"]

        shared_context = state.get("shared_context", {})
        citation_map = shared_context.get("citations", {})
        next_citation_number = shared_context.get("next_citation_number", 1)
        citation_pattern = re.compile(r'\[CITE:\s*(.*?)]\((.*?)\)')

        def replace_and_update_map(match):
            nonlocal next_citation_number
            title, url = match.group(1).strip(), match.group(2).strip()
            if url not in citation_map:
                citation_map[url] = {'title': title, 'number': next_citation_number}
                next_citation_number += 1
            return f"[{citation_map[url]['number']}]({url})"

        processed_content = citation_pattern.sub(replace_and_update_map, raw_content)

        item["content"] = processed_content
        item["status"] = "completed"
        item["execution_log"].append(f"第 {item['attempt_count']} 次撰写完成。")
        writing_plan[item_index] = item
        shared_context['citations'] = citation_map
        shared_context['next_citation_number'] = next_citation_number

        return {"writing_plan": writing_plan, "shared_context": shared_context}

    def call_final_assembler(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- [进入 写作子图: final_assembler (总编辑) 节点] ---")
        writing_plan = state.get("writing_plan", [])
        full_draft = "\n\n---\n\n".join(
            [item['content'] for item in sorted(writing_plan, key=lambda x: x.get('item_id'))])

        citation_map = state.get("shared_context", {}).get("citations", {})
        final_sources_list = []

        if not citation_map:
            return {"final_answer": f"# {state['input']}\n\n" + full_draft, "final_sources": []}

        sorted_citations = sorted(citation_map.items(), key=lambda item: item[1]['number'])

        reference_list_text = ["\n\n---\n\n## 引用来源"]
        for url, data in sorted_citations:
            number = data['number']
            title = data['title']
            reference_list_text.append(f"[{number}] [{title}]({url})")
            final_sources_list.append({"number": number, "title": title, "url": url})

        final_report = f"# {state['input']}\n\n" + full_draft + "\n".join(reference_list_text)

        logger.info("最终报告整合完成，并已生成全局引用列表。")
        return {"final_answer": final_report, "final_sources": final_sources_list}

    def route_writing_flow(self, state: AgentState) -> str:
        if state.get("supervisor_decision") == "SYNTHESIZE": return "final_assembler"
        if not state.get("writing_plan"): return "planner"
        if state.get("current_plan_item_id"): return "writer_agent"
        return END

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("planner", self.call_planner)
        workflow.add_node("writer_agent", self.call_writer_agent)
        workflow.add_node("chapter_summarizer", call_chapter_summarizer)
        workflow.add_node("final_assembler", self.call_final_assembler)
        workflow.set_conditional_entry_point(self.route_writing_flow)
        workflow.add_edge("planner", END)
        workflow.add_edge("writer_agent", "chapter_summarizer")
        workflow.add_edge("chapter_summarizer", END)
        workflow.add_edge("final_assembler", END)
        return workflow.compile()

    def get_graph(self) -> CompiledStateGraph:
        return self.graph
