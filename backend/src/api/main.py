import json
import uuid
from typing import Dict, Any, AsyncGenerator

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from backend.src.config.logging_config import get_logger
from backend.src.config.settings import settings
from backend.src.graphs.deepsearch_graph import DeepSearchGraph
from backend.src.schemas.graph_state import AgentState

logger = get_logger(__name__)

# --- 初始化 ---
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="DeepSearch Advanced Agent Backend API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, 'CORS_ORIGINS') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

deep_search_graph = DeepSearchGraph()
graph_app = deep_search_graph.get_app()
session_states: Dict[str, AgentState] = {}


# --- 辅助函数 ---
def _create_initial_state(user_message: str) -> AgentState:
    """为新会话创建一个符合新架构的初始 AgentState。"""
    return {
        "input": user_message, "chat_history": [HumanMessage(content=user_message)],
        "research_plan": [], "writing_plan": [], "shared_context": {},
        "research_results": [], "final_answer": "", "final_sources": [],
        "completed_chapters_count": 0,  # 初始化章节计数器
        "current_plan_item_id": None, "supervisor_decision": "",
        "intermediate_steps": [], "step_count": 0, "error_log": []
    }


def _generate_log_from_step(step_output: Dict[str, Any], current_task_id: str | None) -> Dict[str, Any] | None:
    """从 LangGraph 的流式输出中解析并生成对前端友好的日志。"""
    if not isinstance(step_output, dict) or not step_output: return None
    step_name = list(step_output.keys())[0]
    output = step_output[step_name]
    if not isinstance(output, dict): return None

    log_entry = {"step_name": step_name, "output": ""}
    if step_name == "supervisor":
        decision = output.get("supervisor_decision")
        target_id = output.get("current_plan_item_id")
        log_entry["step_name"] = "总指挥"
        log_entry[
            "output"] = f"决策：执行 '{decision}'，目标任务：'{target_id}'" if target_id else f"决策：下一步行动 '{decision}'"
        return {"log": log_entry}

    if "research_plan" in output or "writing_plan" in output:
        plan_name = "research_plan" if "research_plan" in output else "writing_plan"
        plan = output.get(plan_name, [])
        item_id = current_task_id
        current_item = next((item for item in plan if item["item_id"] == item_id), None)
        if not current_item:
            if plan and any(item.get('status') == 'pending' for item in plan):
                plan_type = "研究" if plan_name == "research_plan" else "写作"
                log_entry["step_name"] = "规划师"
                log_entry["output"] = f"{plan_type}计划已生成，包含 {len(plan)} 个步骤。"
                return {"log": log_entry}
            return None

        status = current_item.get("status")
        description = current_item.get("description")
        if status == "completed":
            log_entry["step_name"] = "评估员" if plan_name == "research_plan" else "总结员"
            log_entry["output"] = f"任务 '{item_id}' 已完成。\n- 描述: {description}"
            return {"log": log_entry}

        elif status == "in_progress":
            log_entry["step_name"] = "研究员" if plan_name == "research_plan" else "写手"
            log_entry["output"] = f"开始执行任务 '{item_id}'...\n- 描述: {description}"
            return {"log": log_entry}
    return None


async def _stream_completed_chapter_content(state: AgentState, step_output) -> AsyncGenerator[str, None]:
    """
    检查是否有新完成的章节，并生成用于流式传输的事件字符串。
    这是一个独立的、可重构的函数，专门负责发送写作内容。
    """
    step_name = list(step_output.keys())[0]
    writing_plan = step_output[step_name]
    chapters_sent_count = state.get("completed_chapters_count", 0)

    if len(writing_plan) > chapters_sent_count:
        newly_completed_chapter = writing_plan[chapters_sent_count]

        logger.info(
            f"--> [SENDING CHAPTER TO FRONTEND]: 检测到新完成的章节 '{newly_completed_chapter['item_id']}'，准备发送...")

        separator = "" if chapters_sent_count == 0 else "\n\n---\n\n"
        content_to_send = separator + newly_completed_chapter.get("content", "")

        chapter_payload = json.dumps({'content': content_to_send}, ensure_ascii=False)
        yield f"event: chat_content_update\ndata: {chapter_payload}\n\n"

        # 更新状态，表示我们已经发送了这一章
        state["completed_chapters_count"] += 1


# --- API 端点 ---
@app.get("/")
async def read_root():
    return {"message": "DeepSearch Advanced API is running!"}


@app.post("/api/v1/chat/stream")
async def chat_stream(request: Request) -> StreamingResponse:
    """处理聊天请求，并以 Server-Sent Events (SSE) 的方式流式返回代理的完整思考过程和最终结果。"""
    try:
        data = await request.json()
        user_message = data.get("message")
        session_id = data.get("session_id", str(uuid.uuid4()))
        if not user_message:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message content cannot be empty.")
        current_state = _create_initial_state(user_message)

        async def event_generator() -> AsyncGenerator[str, None]:
            final_state = None
            current_task_id_for_logging = None

            try:
                config = {"recursion_limit": 100}
                async for event in graph_app.astream_events(current_state, config=config, version="v2"):
                    kind = event["event"]
                    event_name = event['name']

                    if kind == "on_chain_stream":
                        chunk = event["data"]["chunk"]

                        if event_name == "chapter_summarizer":
                            current_full_state = session_states.get(session_id, current_state)
                            async for chapter_event in _stream_completed_chapter_content(current_full_state, chunk):
                                yield chapter_event
                            session_states[session_id] = current_full_state  # 保存更新后的计数器

                        if "supervisor" in chunk:
                            if "current_plan_item_id" in chunk["supervisor"]:
                                current_task_id_for_logging = chunk["supervisor"]["current_plan_item_id"]

                        parsed_data = _generate_log_from_step(chunk, current_task_id_for_logging)
                        if parsed_data and parsed_data.get("log"):
                            yield f"event: activity_update\ndata: {json.dumps(parsed_data['log'], ensure_ascii=False)}\n\n"

                    elif kind == "on_chain_end":
                        if event["name"] == "DeepSearchGraph":
                            final_state = event["data"]["output"]
                            session_states[session_id] = final_state

                if final_state and final_state.get('final_answer'):
                    final_answer = final_state.get('final_answer', '代理未能生成最终答案。')
                    sources_data = final_state.get('final_sources', [])
                    formatted_sources = [f"[{s['number']}] {s['title']}: {s['url']}" for s in sources_data]
                    final_data = {"answer": final_answer, "sources": formatted_sources}
                    yield f"event: final_response\ndata: {json.dumps(final_data, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error(f"在事件生成期间发生API错误: {e}", exc_info=True)
                error_message = f"后端处理失败: {str(e)}"
                yield f"event: final_response\ndata: {json.dumps({'answer': error_message, 'sources': []}, ensure_ascii=False)}\n\n"
            finally:
                if final_state:
                    session_states[session_id] = final_state
                yield "event: message\ndata: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"在 chat_stream 处理程序中发生API错误: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
