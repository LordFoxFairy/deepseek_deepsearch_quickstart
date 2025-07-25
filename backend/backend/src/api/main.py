import json
import re
import uuid
from typing import Dict

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from backend.src.config.settings import settings
from backend.src.graphs.deepsearch_graph import DeepSearchGraph
from backend.src.schemas.graph_state import AgentState

# 初始化FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="DeepSearch Quickstart Backend API",
)

# 配置CORS (跨域资源共享) 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, 'CORS_ORIGINS') else ["http://localhost:5173",
                                                                                   "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化DeepSearchGraph代理
deep_search_graph = DeepSearchGraph()
graph_app = deep_search_graph.get_app()

# 内存中的会话存储
# 注意：在生产环境中，应将其替换为持久化存储（如Redis或数据库），
# 以确保会话状态在服务器重启或多实例部署时不会丢失。
session_states: Dict[str, AgentState] = {}


@app.get("/")
async def read_root():
    """根路由，用于检查API是否正常运行。"""
    return {"message": "DeepSearch API 运行中！"}


@app.post("/api/v1/chat/stream")
async def chat_stream(request: Request) -> StreamingResponse:
    """
    处理聊天请求并以流式方式返回代理的响应。
    """
    try:
        data = await request.json()
        user_message = data.get("message")
        session_id = data.get("session_id")

        if not user_message:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="消息内容不能为空。")

        if not session_id:
            session_id = str(uuid.uuid4())
            print(f"生成新的会话ID: {session_id}")

        # 检索或初始化会话的代理状态
        current_state = session_states.get(session_id)
        if not current_state:
            current_state: AgentState = {
                "input": user_message,
                "chat_history": [HumanMessage(content=user_message)],
                "plan": [],
                "current_step": None,
                "planning_attempts_count": 0,
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
            # 为现有会话中的新消息更新状态
            current_state["input"] = user_message
            current_state["chat_history"].append(HumanMessage(content=user_message))
            # 重置与新查询相关的字段
            current_state["plan"] = []
            current_state["current_step"] = None
            current_state["planning_attempts_count"] = 0
            current_state["tool_calls"] = []
            current_state["tool_output"] = None
            current_state["evaluation_results"] = None
            current_state["replan_needed"] = False
            current_state["supervisor_decision"] = None
            current_state["step_count"] = 0
            current_state["consecutive_no_progress_count"] = 0

            print(f"更新会话 {session_id} 的状态。")

        async def event_generator():
            """为前端生成服务器发送事件 (SSE)。"""
            try:
                # 为图设置更高的递归限制，默认为25。
                # 这允许代理有更多步骤来完成复杂任务。
                config = {"recursion_limit": 50}

                async for s in graph_app.astream(current_state, config=config):
                    if "__end__" not in s:
                        step_name = list(s.keys())[0]
                        step_output = s[step_name]

                        try:
                            activity_output_display = ""
                            if isinstance(step_output, dict):
                                if step_name == "executor" and "tool_output" in step_output:
                                    tool_output_str = step_output["tool_output"]
                                    formatted_tool_results = []

                                    search_result_pattern = r"SearchResult\(title='([^']+?)', url='([^']+?)', snippet='([^']+?)'\)"
                                    rag_result_pattern = r"RagResult\(content='([^']+?)', source='([^']+?)'\)"

                                    for sr_match in re.finditer(search_result_pattern, tool_output_str):
                                        title = sr_match.group(1)
                                        url = sr_match.group(2)
                                        formatted_tool_results.append(f"- [{title}]({url})")

                                    for rag_match in re.finditer(rag_result_pattern, tool_output_str):
                                        content_snippet = rag_match.group(1)
                                        source = rag_match.group(2)
                                        formatted_tool_results.append(f"- [{content_snippet[:50]}...]({source})")

                                    if formatted_tool_results:
                                        activity_output_display = f"执行器完成步骤. 工具输出 ({len(formatted_tool_results)} 条结果):\n" + "\n".join(
                                            formatted_tool_results)
                                    elif "工具执行失败" in tool_output_str:
                                        activity_output_display = tool_output_str
                                    else:
                                        activity_output_display = f"执行器完成步骤. 工具输出: {tool_output_str}"
                                elif step_name == "planner" and "plan" in step_output:
                                    activity_output_display = "计划: \n" + "\n".join(
                                        [f"- {step}" for step in step_output["plan"]])
                                elif step_name == "evaluator" and "evaluation_results" in step_output:
                                    activity_output_display = "评估结果: " + step_output["evaluation_results"]
                                elif step_name == "writer" and "final_answer" in step_output:
                                    activity_output_display = "写作完成，等待评审。"
                                    final_data = {
                                        "answer": step_output['final_answer'],
                                        "sources": []
                                    }
                                    yield f"event: final_response\n"
                                    yield f"data: {json.dumps(final_data)}\n\n"
                                elif step_name == "reviewer" and "evaluation_results" in step_output:
                                    activity_output_display = "评审结果: " + step_output["evaluation_results"]
                                elif step_name == "supervisor" and "supervisor_decision" in step_output:
                                    activity_output_display = "主管决策: " + step_output["supervisor_decision"]
                                    if step_output["supervisor_decision"] == "FAIL" and "final_answer" in step_output:
                                        yield f"event: chat_content_update\n"
                                        yield f"data: {json.dumps({'content': step_output['final_answer']})}\n\n"
                                else:
                                    activity_output_display = str(step_output)
                            else:
                                activity_output_display = str(step_output)

                            yield f"event: activity_update\n"
                            yield f"data: {json.dumps({'step_name': step_name, 'output': activity_output_display})}\n\n"
                        except TypeError as e:
                            print(
                                f"警告: 无法序列化步骤 {step_name} 的输出 ({e})。原始输出: {step_output}。跳过活动更新。")

                    else:
                        # 流结束
                        yield "event: message\ndata: [DONE]\n\n"
                        return

            except Exception as e:
                print(f"在事件生成期间发生API错误: {e}")
                error_message_for_frontend = f"后端处理失败: {str(e)}"
                yield "event: message\ndata: [DONE]\n\n"
                yield f"event: final_response\n"
                yield f"data: {json.dumps({'answer': error_message_for_frontend, 'sources': []})}\n\n"
                return

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        print(f"在chat_stream处理程序中发生API错误: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
