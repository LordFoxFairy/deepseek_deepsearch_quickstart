import uuid
from typing import Dict

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from backend.src.config.settings import settings
from backend.src.graphs.deepsearch_graph import DeepSearchGraph
from backend.src.schemas.graph_state import AgentState  # 导入 AgentState
from backend.src.schemas.tool_models import SearchResult, RagResult  # 导入 SearchResult 和 RagResult，用于来源提取

# 初始化 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="DeepSearch Quickstart Backend API",
)

# 配置 CORS 中间件
# 允许来自前端应用的跨域请求
app.add_middleware(
    CORSMiddleware,
    # 允许的来源列表，从 settings 中加载，如果 settings 中没有，则使用默认值
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, 'CORS_ORIGINS') else ["http://localhost:5173",
                                                                                   "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 DeepSearchGraph
deep_search_graph = DeepSearchGraph()
graph_app = deep_search_graph.get_app()

# 用于存储会话状态的字典
# 注意：在生产环境中，这应该替换为持久化的存储，例如 Redis 或数据库，以确保会话状态在服务器重启或多实例部署时不会丢失。
session_states: Dict[str, AgentState] = {}


@app.get("/")
async def read_root():
    """
    根路径，用于检查 API 是否正常运行。
    """
    return {"message": "DeepSearch API 运行中！"}


@app.post("/api/v1/chat/stream")
async def chat_stream(request: Request) -> StreamingResponse:
    """
    处理聊天请求并以流式方式返回 LLM 响应。
    """
    try:
        data = await request.json()
        user_message = data.get("message")
        session_id = data.get("session_id")

        if not user_message:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="消息内容不能为空。")

        # 如果没有提供 session_id，则生成一个新的
        if not session_id:
            session_id = str(uuid.uuid4())
            print(f"生成新的会话ID: {session_id}")

        # 获取或初始化会话状态
        current_state = session_states.get(session_id)
        if not current_state:
            # 初始化 AgentState 的所有字段
            current_state: AgentState = {
                "input": user_message,
                "chat_history": [HumanMessage(content=user_message)],
                "plan": [],
                "current_step": None,
                "tool_calls": [],
                "tool_output": None,
                "intermediate_steps": [],
                "research_results": [],
                "raw_research_content": "",
                "final_answer": None,
                "evaluation_results": None,
                "replan_needed": False,
                "supervisor_decision": None,
                "step_count": 0,
                "consecutive_no_progress_count": 0
            }
            print(f"为会话 {session_id} 初始化新状态。")
        else:
            # 更新现有状态的输入和聊天历史
            current_state["input"] = user_message
            current_state["chat_history"].append(HumanMessage(content=user_message))
            # 重置与新查询相关的状态，但保留历史和研究结果
            current_state["plan"] = []
            current_state["current_step"] = None
            current_state["tool_calls"] = []
            current_state["tool_output"] = None
            current_state["evaluation_results"] = None
            current_state["replan_needed"] = False
            current_state["supervisor_decision"] = None
            current_state["step_count"] = 0  # 新查询，重置步骤计数
            current_state["consecutive_no_progress_count"] = 0  # 新查询，重置无进展计数

            print(f"更新会话 {session_id} 的状态。")

        async def event_generator():
            full_response_content = ""
            sources = []

            # 使用 LangGraph 的 stream 方法
            # Pass the current_state to the graph stream
            async for s in graph_app.astream(current_state):
                # 迭代每个步骤的输出
                if "__end__" not in s:
                    # 获取当前步骤的名称和输出
                    step_name = list(s.keys())[0]
                    step_output = s[step_name]
                    print(f"--- 步骤: {step_name} ---")
                    print(f"输出: {step_output}")

                    # 根据步骤更新前端，并收集最终答案和来源
                    if step_name == "writer" and "final_answer" in step_output and step_output["final_answer"]:
                        # 写作器生成初步答案
                        full_response_content = step_output["final_answer"]
                        yield f"data: {full_response_content}\n\n"  # 立即发送部分答案
                    elif step_name == "synthesizer" and "final_answer" in step_output and step_output["final_answer"]:
                        # 综合器生成最终答案（如果 writer 未使用）
                        full_response_content = step_output["final_answer"]
                        yield f"data: {full_response_content}\n\n"  # 立即发送部分答案
                    elif step_name == "supervisor" and step_output.get("supervisor_decision") == "FAIL":
                        # 如果主管强制终止并标记为失败
                        fail_message = step_output.get("final_answer", "任务未能完成。")
                        full_response_content = fail_message
                        yield f"data: {fail_message}\n\n"
                    # 可以添加更多步骤的实时输出，例如规划、执行日志
                    # yield f"event: {step_name}\ndata: {step_output}\n\n"
                else:
                    # 流程结束，发送最终信号和所有收集到的来源
                    final_state = s["__end__"]
                    # 更新会话状态以备下次使用
                    session_states[session_id] = final_state
                    print(f"会话 {session_id} 流程结束。最终状态已保存。")

                    # 确保发送最终的完整响应和来源
                    final_answer_from_state = final_state.get("final_answer", "未能生成最终答案。")

                    # 从 research_results 中提取所有唯一的 URL 作为来源
                    unique_sources = set()
                    for res in final_state.get("research_results", []):
                        # 检查是否是 Pydantic 模型实例
                        if isinstance(res, SearchResult) and res.url:
                            unique_sources.add(res.url)
                        elif isinstance(res, RagResult) and res.source:
                            unique_sources.add(res.source)
                        # 如果是字典形式（旧格式或某些工具输出），则按键访问
                        elif isinstance(res, dict) and 'url' in res and res['url']:
                            unique_sources.add(res['url'])
                        elif isinstance(res, dict) and 'source' in res and res['source']:
                            unique_sources.add(res['source'])

                    sources = list(unique_sources)

                    yield f"data: [DONE]\n"
                    yield f"event: final_response\n"
                    yield f"data: {final_answer_from_state}\n"
                    yield f"data: {sources}\n\n"  # 发送来源列表
                    break  # 结束流

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        print(f"API 错误: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    # 此块用于在开发环境中直接运行 FastAPI 应用
    # 生产环境通常使用 uvicorn 命令运行
    import uvicorn

    print("正在启动 FastAPI 应用...")
    # 注意：这里硬编码了端口，实际部署时应通过配置管理
    uvicorn.run(app, host="0.0.0.0", port=8000)
