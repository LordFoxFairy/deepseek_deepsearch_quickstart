# DeepSearch AI Agent 🚀

**DeepSearch AI Agent** 是一个基于 LangGraph 和 FastAPI 构建的先进 AI 代理项目。它通过一个复杂的、自校正的流程来处理用户查询，该流程包括规划、工具执行、评估和答案生成。前端使用 React 和 Vite 构建，提供了一个流畅的聊天界面和实时的代理活动时间线。

## ✨ 功能

- **高级代理架构**: 使用 LangGraph 的状态图（StateGraph）构建，实现了一个包含规划、执行、评估、写作和评审的完整循环，赋予代理深度思考和自我修正的能力。
- **工具使用**:
  - **网络搜索**: 集成了 `googlesearch`，能够从互联网获取最新信息来回答问题。
  - **检索增强生成 (RAG)**: 利用 `LlamaIndex` 从内部知识库或动态索引的文档中检索信息，提供更具深度的答案。
- **实时流式响应**: 后端采用 FastAPI 的 `StreamingResponse`，前端可以实时接收和显示代理的思考过程和最终答案。
- **交互式前端**:
  - 使用 Vite + React + TypeScript 构建，提供快速的开发体验和类型安全。
  - 采用 Tailwind CSS 和 `shadcn/ui` 构建，界面美观且响应式。
  - **代理活动时间线**: 实时展示代理的每一步决策（如规划、调用工具、评估结果），使用户能够直观地了解 AI 的“思考”过程。
- **灵活的配置**: 通过 `.env` 文件和 Pydantic 设置类管理 API 密钥和模型配置，轻松切换不同的 LLM 服务（如 DeepSeek, DashScope 等）。

## 📐 架构

项目采用前后端分离的架构：

1. **Frontend (React + Vite)**:
   - 一个现代化的单页面应用 (SPA)。
   - 负责用户交互、消息展示和与后端 API 的通信。
   - 通过 Server-Sent Events (SSE) 接收来自后端的流式数据。
2. **Backend (FastAPI)**:
   - 提供一个 `/api/v1/chat/stream` API 端点用于处理聊天请求。
   - 使用会话（Session）来维护多轮对话的状态。
   - 核心是一个 **`DeepSearchGraph`** 实例，这是一个用 `LangGraph` 定义的状态机。

### LangGraph 代理流程

代理的执行流程由一个主管（Supervisor）节点控制，该节点根据当前状态决定将任务路由到哪个子代理：

```
graph TD
    A[用户请求] --> B(Supervisor);
    B --> C{有计划吗？};
    C -- 否 --> D[Planner: 制定计划];
    D --> B;
    C -- 是 --> E{计划执行完了吗？};
    E -- 否 --> F[Executor: 执行工具];
    F --> G[Evaluator: 评估结果];
    G -- 结果不好 --> D;
    G -- 结果够好 --> B;
    E -- 是 --> H[Writer: 撰写报告];
    H --> I[Reviewer: 评审报告];
    I -- 不通过 --> D;
    I -- 通过 --> J[FINISH: 生成最终答案];
    J --> K[用户];

    style F fill:#f9f,stroke:#333,stroke-width:2px;
    style D fill:#ccf,stroke:#333,stroke-width:2px;
    style G fill:#cfc,stroke:#333,stroke-width:2px;
    style H fill:#ffc,stroke:#333,stroke-width:2px;
    style I fill:#fca,stroke:#333,stroke-width:2px;
```

## 🛠️ 技术栈

|              | **技术**                                                     |
| ------------ | ------------------------------------------------------------ |
| **后端**     | Python, FastAPI, Langchain, LangGraph, LlamaIndex, Pydantic, Uvicorn |
| **前端**     | React, TypeScript, Vite, Tailwind CSS, shadcn/ui, lucide-react |
| **AI 模型**  | DeepSeek (默认), DashScope (用于嵌入), 兼容 OpenAI 的各类模型 |
| **核心依赖** | `langchain-core`, `langgraph`, `llama-index`, `fastapi`, `react`, `vite` |

## 🚀 快速开始

### 1. 克隆仓库

```
git clone <your-repo-url>
cd deepseek_deepsearch_quickstart
```

### 2. 后端设置

a. **创建并激活虚拟环境** (推荐):

```
python -m venv venv
source venv/bin/activate  # on Windows, use `venv\Scripts\activate`
```

b. **安装依赖**:

```
cd backend
pip install -r requirements.txt
```

c. 配置环境变量:

在 backend/src/ 目录下，复制 .env 文件。

```
cp backend/src/.env.example backend/src/.env
```

然后编辑 `.env` 文件，填入你的 API 密钥：

```
# .env

# 用于 RAG 嵌入
DASH_SCOPE_API_KEY=sk-your-dashscope-api-key
DASH_SCOPE_EMBEDDING_MODEL=text-embedding-v1

# 用于 LLM 推理
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
DEEPSEEK_CHAT_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### 3. 前端设置

a. **安装依赖**:

```
cd frontend
npm install
```

### 4. 运行应用

a. 启动后端服务器:

在 backend 目录下运行：

```
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

服务器将在 `http://localhost:8000` 启动。

b. 启动前端开发服务器:

在 frontend 目录下运行：

```
npm run dev
```

应用将在 `http://localhost:5173` 启动。

现在，打开浏览器访问 `http://localhost:5173` 即可开始与 DeepSearch AI Agent 交互！

## 📂 项目结构

```
deepseek_deepsearch_quickstart/
├── backend/
│   ├── src/
│   │   ├── api/
│   │   │   └── main.py           # FastAPI 应用主入口
│   │   ├── config/
│   │   │   └── settings.py       # Pydantic 配置管理
│   │   ├── graphs/
│   │   │   └── deepsearch_graph.py # LangGraph 核心逻辑
│   │   ├── llms/
│   │   │   └── openai_llm.py     # LLM 模型加载
│   │   ├── prompts/              # 存放所有 Prompt
│   │   ├── schemas/              # Pydantic 数据模型
│   │   ├── services/             # 外部服务封装 (搜索, LlamaIndex)
│   │   ├── tools/                # Langchain 工具定义
│   │   └── .env                  # 环境变量
│   └── requirements.txt          # Python 依赖
│
├── frontend/
│   ├── src/
│   │   ├── components/           # UI 组件
│   │   ├── features/             # 核心功能模块 (如聊天)
│   │   ├── lib/                  # 工具函数
│   │   ├── App.tsx               # 应用主组件
│   │   └── main.tsx              # 应用入口
│   ├── package.json              # Node.js 依赖和脚本
│   └── vite.config.ts            # Vite 配置文件
│
└── README.md                     # 本文档
```

## 📜 API 端点

- `GET /`: 健康检查端点。

- `POST /api/v1/chat/stream`: 接收聊天消息并以流式方式返回响应。

  - **请求体**:

    ```
    {
      "message": "你的问题是什么？",
      "session_id": "一个唯一的用户会话 ID"
    }
    ```

  - **响应**: 一个 `text/event-stream` 流，包含多种类型的事件，如 `activity_update` 和 `final_response`。