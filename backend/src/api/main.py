import json
from typing import AsyncGenerator, Set, Dict, Any

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from backend.src.config.logging_config import get_logger
from backend.src.config.settings import settings
from backend.src.graphs.deepsearch_graph import DeepSearchGraph
from backend.src.schemas.graph_state import AgentState

logger = get_logger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version="2.0.0",
    description="DeepSearch Advanced Agent Backend API (V2 - Phased Execution)",
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


def _create_initial_state(user_message: str) -> AgentState:
    """创建图的初始状态，确保所有字段都被初始化。"""
    return {
        "input": user_message, "chat_history": [HumanMessage(content=user_message)],
        "overall_outline": None, "plan": [], "final_answer": "", "final_sources": [],
        "current_plan_item_id": None, "supervisor_decision": "", "step_count": 0,
        "error_log": [], "shared_context": {"citations": {}, "next_citation_number": 1},
        "next_step_index": 0,
    }


def _format_sse(event_type: str, data: Dict[str, Any]) -> str:
    """将数据格式化为 Server-Sent Event (SSE) 字符串。"""
    json_data = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {json_data}\n\n"


@app.post("/api/v1/chat/stream", response_class=StreamingResponse)
async def chat_stream(request: Request):
    try:
        body = await request.json()
        user_message = body.get("message", "")
        if not user_message:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="消息不能为空")

        initial_state = _create_initial_state(user_message)

        async def event_generator() -> AsyncGenerator[str, None]:
            total_research_tasks = 0
            total_writing_tasks = 0
            completed_research_tasks = 0
            completed_writing_tasks = 0
            plan_initialized = False
            sent_chapter_ids: Set[str] = set()

            try:
                config = {"recursion_limit": 50}
                async for event in graph_app.astream_events(initial_state, version="v1", config=config):
                    event_name = event["event"]

                    # 不再发送底层的、对用户不够直观的日志事件
                    # if event_name.startswith("on_") and event.get("name"):
                    #      log_message = f"Event: '{event_name}' for node '{event['name']}'"
                    #      yield _format_sse("log", {"message": log_message})

                    if event_name.endswith("_end"):
                        node_name = event["name"]
                        output_state = event["data"].get("output")

                        if not isinstance(output_state, dict): continue

                        current_plan = output_state.get("plan", [])

                        if node_name == 'planner' and not plan_initialized:
                            if current_plan:
                                total_research_tasks = len([t for t in current_plan if t['task_type'] == 'RESEARCH'])
                                total_writing_tasks = len([t for t in current_plan if t['task_type'] == 'WRITING'])
                                plan_initialized = True
                                logger.info(
                                    f"计划初始化: {total_research_tasks} 个研究任务, {total_writing_tasks} 个写作任务。")

                        if node_name == 'research_executor':
                            completed_research_tasks += 1
                            task_id = event["data"]["input"].get("current_plan_item_id")
                            task = next((t for t in current_plan if t["item_id"] == task_id), None)
                            description = task['description'] if task else "正在研究..."

                            progress_payload = {
                                "type": "research", "current": completed_research_tasks,
                                "total": total_research_tasks, "description": description
                            }
                            yield _format_sse("progress", progress_payload)

                        if node_name == 'writing_executor':
                            completed_writing_tasks += 1
                            task_id = event["data"]["input"].get("current_plan_item_id")
                            task = next((t for t in current_plan if t["item_id"] == task_id), None)

                            if task and task_id not in sent_chapter_ids:
                                progress_payload = {
                                    "type": "writing", "current": completed_writing_tasks,
                                    "total": total_writing_tasks, "description": task['description']
                                }
                                yield _format_sse("progress", progress_payload)

                                chapter_payload = {
                                    "item_id": task_id, "title": task.get("description"),
                                    "content": task.get("content", "")
                                }
                                yield _format_sse("chapter", chapter_payload)
                                sent_chapter_ids.add(task_id)

                        if node_name == "final_assembler":
                            if output_state.get('final_answer'):
                                yield _format_sse("references", {"content": output_state['final_answer']})
                            if output_state.get('final_sources'):
                                yield _format_sse("sources", {"sources": output_state['final_sources']})

                logger.info("--- 图执行流程结束 ---")

            except Exception as e:
                logger.error(f"在事件生成期间发生错误: {e}", exc_info=True)
                yield _format_sse("error", {"error": f"后端处理失败: {str(e)}"})
            finally:
                yield "event: end\ndata: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"在 /chat/stream 端点发生严重错误: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
