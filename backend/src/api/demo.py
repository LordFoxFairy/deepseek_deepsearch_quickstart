import asyncio
import json
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# 创建一个 FastAPI 应用实例
app = FastAPI(title="Stress Test Mock AI Agent Backend")

# 添加CORS中间件，允许所有来源的跨域请求，方便前端调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _format_sse(event_type: str, data: Dict[str, Any]) -> str:
    """将数据格式化为 Server-Sent Event (SSE) 字符串。"""
    json_data = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {json_data}\n\n"


# 用于生成海量测试数据的文本块
LOREM_IPSUM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "
    "这是一个用于测试长文本渲染的中文段落。这个段落会重复多次以模拟大量数据一次性推送到前端的场景。 "
    "$$ E=mc^2 $$ "
    "这是另一个包含数学公式的段落，以确保Katex渲染也能被正确测试。 [ref:https://example.com/stress-test]\n\n"
)


@app.post("/api/v1/chat/stream")
async def chat_stream(request: Request):
    """
    模拟一个一次性发送超大量数据的后端流式接口，用于压力测试前端UI。
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. 发送一个初始进度事件，让前端知道任务已开始
            yield _format_sse("progress", {
                "type": "writing",
                "current": 1,
                "total": 1,
                "description": "正在生成超大量测试数据..."
            })
            await asyncio.sleep(1)  # 短暂延迟，确保前端UI已准备好接收

            # 2. 生成并一次性发送海量数据
            # 将上面的文本块重复50次，创建一个非常长的字符串
            massive_content = LOREM_IPSUM * 500

            chapter_payload = {
                "item_id": "stress_test_chapter",
                "title": "压力测试章节：海量数据渲染",
                "content": massive_content
            }
            print("准备发送海量数据...")
            yield _format_sse("chapter", chapter_payload)
            print("海量数据已发送。")

        except Exception as e:
            error_payload = {"error": f"模拟服务器错误: {str(e)}"}
            yield _format_sse("error", error_payload)
        finally:
            # 3. 发送结束信号
            yield "event: end\ndata: [DONE]\n\n"
            print("压力测试数据流发送完毕。")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    print("启动压力测试模拟后端服务器，访问 http://localhost:8000")
    print("现在您可以启动前端应用，并向此服务器发送请求来调试UI的气泡渲染问题。")
    uvicorn.run(app, host="0.0.0.0", port=8000)
