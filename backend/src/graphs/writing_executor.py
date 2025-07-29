import re
from typing import List, Dict, Any, Optional, Tuple

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool

from backend.src.config.logging_config import get_logger
from backend.src.llms.openai_llm import get_chat_model
from backend.src.prompts.writer_prompts import WRITER_PROMPT
from backend.src.schemas.graph_state import AgentState, PlanItem
from backend.src.services.llama_index_service import llama_index_service

llm = get_chat_model()
logger = get_logger(__name__)


def _find_plan_item(plan: List[PlanItem], item_id: str) -> Tuple[Optional[PlanItem], int]:
    """在计划列表中根据ID查找任务项及其索引。"""
    for i, item in enumerate(plan):
        if item.get("item_id") == item_id:
            return item, i
    return None, -1


def _create_rag_tool_for_writing(current_item: PlanItem) -> Tool:
    """为写作任务创建范围化的 RAG 工具。"""
    dependency_ids = current_item.get("dependencies", [])
    if not dependency_ids:
        logger.warning(f"写作任务 '{current_item['description']}' 没有指定研究依赖。RAG 将在整个知识库中进行。")
        return Tool(
            name="rag_tool",
            func=lambda query: llama_index_service.query_index_with_metadata_filter(query, "research_task_id", []),
            description="检索关于特定主题的详细信息和数据。"
        )

    logger.info(f"为写作任务 '{current_item['description']}' 创建范围化 RAG 工具，依赖: {dependency_ids}")

    def scoped_query(query: str) -> str:
        return llama_index_service.query_index_with_metadata_filter(
            query,
            filter_key="research_task_id",
            filter_values=dependency_ids
        )

    return Tool(
        name="rag_tool",
        func=scoped_query,
        description="从相关的研究资料中，检索关于特定主题的详细信息和数据。"
    )


def _process_citations_and_update_state(
        raw_content: str,
        current_item: PlanItem,
        updated_plan: List[PlanItem],
        item_index: int,
        shared_context: Dict[str, Any]
) -> Dict[str, Any]:
    """处理LLM生成的文本中的[ref:url]引用，并更新状态。"""
    logger.info(f"开始为任务 '{current_item['description']}' 处理引用。")
    citation_map = shared_context.get("citations", {})
    next_citation_number = shared_context.get("next_citation_number", 1)

    citation_pattern = re.compile(r'\[ref:(https?://[^\s\]]+)\]')

    def replace_and_update_map(match):
        nonlocal next_citation_number
        url = match.group(1)
        if url not in citation_map:
            source_node = llama_index_service.get_document_by_source_url(url)
            title = source_node.metadata.get('title', '未知标题') if source_node else '未知标题'

            citation_map[url] = {
                "number": next_citation_number,
                "title": title,
                "url": url
            }
            logger.info(f"新增引用: [ {next_citation_number} ] {title} ({url})")
            next_citation_number += 1

        number = citation_map[url]['number']
        return f"[{number}]({url})"

    processed_content = citation_pattern.sub(replace_and_update_map, raw_content)

    updated_plan[item_index]["content"] = processed_content
    updated_plan[item_index]["status"] = "completed"
    shared_context['citations'] = citation_map
    shared_context['next_citation_number'] = next_citation_number

    return {"plan": updated_plan, "shared_context": shared_context}


async def execute_writing_task(state: AgentState) -> Dict[str, Any]:
    """执行单个写作任务，为 Agent 提供完整的上下文。"""
    current_item_id = state.get("current_plan_item_id")
    if not current_item_id:
        raise ValueError("execute_writing_task: current_plan_item_id 缺失。")

    plan = state["plan"]
    item, item_index = _find_plan_item(plan, current_item_id)
    if not item:
        raise ValueError(f"execute_writing_task: 未找到ID为 {current_item_id} 的任务。")

    logger.info(f"正在处理写作任务: '{item['description']}'")

    rag_tool = _create_rag_tool_for_writing(item)
    tools = [rag_tool]

    agent = create_openai_tools_agent(llm, tools, WRITER_PROMPT)

    # 为 Agent Executor 添加 max_iterations 参数，防止无限循环
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=7  # 设置每个写作任务中，工具调用的最大次数
    )

    overall_outline = state.get("overall_outline", "未提供总大纲。")
    all_writing_tasks = [t for t in plan if t.get("task_type") == "WRITING"]
    all_chapter_summaries_list = [
        f"章节目标: {t.get('description', '无描述')}\n核心内容摘要: {t.get('content', '摘要不可用。')}" for t in
        all_writing_tasks]
    all_chapter_summaries = "\n\n---\n\n".join(all_chapter_summaries_list)

    previous_chapter_content = "这是报告的第一章。"
    current_writing_task_index = next(
        (i for i, task in enumerate(all_writing_tasks) if task.get("item_id") == current_item_id), -1)

    if current_writing_task_index > 0:
        previous_task_id = all_writing_tasks[current_writing_task_index - 1]['item_id']
        prev_item, _ = _find_plan_item(plan, previous_task_id)
        if prev_item and prev_item.get('status') == 'completed':
            previous_chapter_content = prev_item.get('content', "前一章内容为空。")

    agent_input = {
        "input": state.get("input"),
        "task_description": item["description"],
        "overall_outline": overall_outline,
        "all_chapter_summaries": all_chapter_summaries,
        "previous_chapter_content": previous_chapter_content,
        "revision_notes": "",
    }

    response = await agent_executor.ainvoke(agent_input)

    raw_content = response['output']
    updated_plan = [p.copy() for p in plan]
    shared_context = state.get("shared_context", {}).copy()

    return _process_citations_and_update_state(
        raw_content, item, updated_plan, item_index, shared_context
    )


async def final_assembler(state: AgentState) -> Dict[str, Any]:
    """最终整合 - 生成格式更优美的参考文献列表。"""
    logger.info("--- [阶段 5] 进入 final_assembler 节点 (仅生成引用列表) ---")
    citation_map = state.get("shared_context", {}).get("citations", {})

    if not citation_map:
        return {"final_answer": "", "final_sources": []}

    sorted_citations = sorted(citation_map.values(), key=lambda c: c['number'])

    reference_list_lines = ["\n\n---\n\n## 参考文献"]
    for data in sorted_citations:
        reference_list_lines.append(
            f"[{data['number']}] {data.get('title', '未知标题')}. Available: {data['url']}"
        )

    final_reference_text = "\n".join(reference_list_lines)
    return {"final_answer": final_reference_text, "final_sources": sorted_citations}
