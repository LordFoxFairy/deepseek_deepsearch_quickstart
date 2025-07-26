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
            final_state = None
            try:
                config = {"recursion_limit": 50}

                async for s in graph_app.astream(current_state, config=config):
                    if "__end__" not in s:
                        step_name = list(s.keys())[0]
                        step_output = s[step_name]

                        try:
                            ### 修改点 1: 增强活动更新的解析逻辑 ###
                            activity_output_display = ""
                            # 为 supervisor 节点生成日志
                            if step_name == "supervisor":
                                decision = step_output.get("supervisor_decision", "决策中...")
                                activity_output_display = f"主管决策: {decision}"

                            # 为 research_flow 子图生成更详细的日志
                            elif step_name == "research_flow":
                                eval_result = step_output.get("evaluation_results", "无")
                                num_results = len(step_output.get("research_results", []))
                                activity_output_display = (
                                    f"研究流程完成。\n"
                                    f"- 收集到 {num_results} 条结果。\n"
                                    f"- 评估结果: {eval_result}"
                                )

                            # 为 writing_flow 子图生成更详细的日志
                            elif step_name == "writing_flow":
                                review_result = step_output.get("evaluation_results", "无")
                                activity_output_display = f"写作流程完成。\n- 评审结果: {review_result}"

                                ### 修改点 2: 引入实时内容更新事件 ###
                                # 当写作流程结束时，报告初稿已生成，立即将其发送给前端。
                                if 'final_answer' in step_output and step_output['final_answer']:
                                    print("检测到写作流程已生成报告，向前端发送实时内容更新。")
                                    yield f"event: chat_content_update\n"
                                    yield f"data: {json.dumps({'content': step_output['final_answer']})}\n\n"

                            else:
                                activity_output_display = f"进入未知节点: {step_name}"

                            yield f"event: activity_update\n"
                            yield f"data: {json.dumps({'step_name': step_name, 'output': activity_output_display})}\n\n"
                        except Exception as e:
                            print(f"警告: 解析步骤 {step_name} 的输出时出错 ({e})。")

                    else:
                        final_state = s["__end__"]
                        break

                # 在流程结束后，发送最终答案
                if final_state:
                    print("流程结束，正在向前端发送最终答案。")
                    final_answer = final_state.get('final_answer', '代理未能生成最终答案。')

                    sources = []  # 来源提取逻辑可在此处实现

                    final_data = {
                        "answer": final_answer,
                        "sources": sources
                    }
                    yield f"event: final_response\n"
                    yield f"data: {json.dumps(final_data)}\n\n"

            except Exception as e:
                print(f"在事件生成期间发生API错误: {e}")
                error_message_for_frontend = f"后端处理失败: {str(e)}"
                yield f"event: final_response\n"
                yield f"data: {json.dumps({'answer': error_message_for_frontend, 'sources': []})}\n\n"

            finally:
                # 保存更新后的会话状态
                if final_state:
                    session_states[session_id] = final_state
                    print(f"已为会话 {session_id} 保存最终状态。")

                # 发送流结束信号
                yield "event: message\ndata: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        print(f"在chat_stream处理程序中发生API错误: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
