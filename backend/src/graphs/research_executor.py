import asyncio
from typing import Dict, Any, List, Optional, Tuple

from backend.src.config.logging_config import get_logger
from backend.src.llms.openai_llm import get_chat_model
from backend.src.schemas.graph_state import AgentState, PlanItem
from backend.src.services.llama_index_service import llama_index_service
from backend.src.tools.search_tools import search_tool

llm = get_chat_model()
logger = get_logger(__name__)


def _find_plan_item(plan: List[PlanItem], item_id: str) -> Tuple[Optional[PlanItem], int]:
    """在计划列表中根据ID查找任务项及其索引。"""
    for i, item in enumerate(plan):
        if item.get("item_id") == item_id:
            return item, i
    return None, -1


async def execute_research_task(state: AgentState) -> Dict[str, Any]:
    """
    执行单个研究任务。
    职责: 1. 执行搜索。 2. 将原始结果存入知识库。 3. 将原始片段存入 plan。
    """
    logger.info("--- [执行研究任务] ---")
    current_item_id = state.get("current_plan_item_id")
    if not current_item_id:
        logger.error("execute_research_task: 未提供 current_plan_item_id，这是一个逻辑错误。")
        return {"error_log": [{"node": "execute_research_task", "error": "current_plan_item_id 缺失"}]}

    plan = state["plan"]
    item, item_index = _find_plan_item(plan, current_item_id)
    if not item:
        error_msg = f"未找到ID为 {current_item_id} 的任务。"
        logger.error(error_msg)
        return {"error_log": [{"node": "execute_research_task", "error": error_msg}]}

    logger.info(f"开始研究: '{item['description']}'")
    updated_plan = [p.copy() for p in plan]
    current_item = updated_plan[item_index]
    current_item['status'] = 'in_progress'

    try:
        # 1. 执行搜索
        search_results = await asyncio.to_thread(search_tool.invoke, {"query": current_item['description']})
        current_item['execution_log'].append(f"成功执行搜索，获得 {len(search_results)} 条结果。")

        # 2. 将结果存入 LlamaIndex (用于 RAG)
        metadata = {"research_task_id": current_item_id}
        llama_index_service.add_search_results_to_index(search_results, metadata)
        logger.info(
            f"已将 {len(search_results)} 条搜索结果存入知识库，并打上标签: 'research_task_id: {current_item_id}'")

        # 直接存储原始片段
        snippets = [f"来源: {res.url}\n标题: {res.title}\n片段: {res.snippet}"
                    for res in search_results if hasattr(res, 'snippet') and res.snippet]

        if not snippets:
            logger.warning(f"研究任务 '{item['description']}' 没有找到可用的网页片段。")
            raw_content = "没有找到相关信息。"
        else:
            # 将所有原始片段用分隔符连接起来，作为此研究任务的产出
            raw_content = "\n\n---\n\n".join(snippets)

        # 3. 更新任务状态和内容 (内容现在是原始片段集合)
        current_item['content'] = raw_content
        current_item['status'] = 'completed'
        current_item['execution_log'].append("研究任务完成，已存储原始网页片段。")
        logger.info(f"研究任务 '{item['description']}' 已完成，原始片段已存储。")

        return {"plan": updated_plan}

    except Exception as e:
        error_msg = f"执行研究任务 '{item['description']}' 时出错: {e}"
        logger.error(error_msg, exc_info=True)
        current_item['status'] = 'failed'
        current_item['execution_log'].append(error_msg)
        return {
            "plan": updated_plan,
            "error_log": [{"node": "execute_research_task", "error": str(e)}]
        }
