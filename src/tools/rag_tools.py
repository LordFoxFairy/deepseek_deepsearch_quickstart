from langchain_core.tools import StructuredTool
from backend.src.services.llama_index_service import LlamaIndexService
from backend.src.schemas.tool_models import RagToolInput, RagResult
from typing import List, Any  # 导入 Any 用于 kwargs 类型提示

# 初始化 LlamaIndex 服务实例
llama_index_service = LlamaIndexService()


def _run_rag_query(**kwargs: Any) -> List[RagResult]:
    """
    执行 RAG（检索增强生成）查询，从知识库中检索相关信息。
    这是一个包装函数，用于将 LlamaIndexService 的检索结果适配为工具的输出。

    Args:
        **kwargs: 包含查询字符串和要检索文档数量的关键字参数。
                  这些参数将被用于创建 RagToolInput 实例。

    Returns:
        List[RagResult]: 检索到的文档内容和来源列表。
    """
    try:
        # 从 kwargs 创建 RagToolInput 实例
        tool_input = RagToolInput(**kwargs)
    except Exception as e:
        print(f"创建 RagToolInput 失败: {e}. 传入的参数: {kwargs}")
        # 可以选择抛出异常或返回空列表
        raise ValueError(f"无效的工具输入: {e}")

    # 从 Pydantic 输入模型中解构参数
    query = tool_input.query
    top_k = tool_input.top_k

    # 在实际使用 RAG 工具前，需要确保 LlamaIndex 已经构建了索引。
    # 这里我们假设索引已经通过某种方式（例如，在应用启动时或通过特定 API）被填充。
    return llama_index_service.retrieve(query=query, top_k=top_k)


# 定义 Langchain RAG 工具
# 使用 StructuredTool 来确保 Pydantic 输入模型被正确处理。
rag_tool = StructuredTool(
    name="rag_tool",
    description="用于从内部知识库或文档中检索相关信息的工具。输入是一个查询字符串和可选的返回文档数量。",
    func=_run_rag_query,
    args_schema=RagToolInput  # 使用 Pydantic 模型作为工具的输入 Schema
)

if __name__ == "__main__":
    # 此块用于测试 rag_tools.py
    print("正在测试 rag_tools.py...")

    # 模拟构建一个简单的 LlamaIndex 索引
    # 在实际应用中，这些文档可能来自数据库、文件系统或网页抓取
    sample_documents = [
        "人工智能（AI）在2024年取得了显著进展，特别是在大型语言模型（LLM）和多模态AI方面。",
        "具身智能是AI的另一个热门方向，旨在让AI系统能够在物理世界中进行交互和学习。",
        "生成式AI的应用越来越广泛，从艺术创作到代码生成都有体现。",
        "量子计算虽然仍处于早期阶段，但其潜力巨大，未来可能彻底改变计算方式。",
        "边缘AI使得AI模型能够在设备本地运行，减少了对云端的依赖，提高了隐私和响应速度。"
    ]
    # 为每个文档提供一个模拟的 ID 或 URL
    sample_doc_ids = [
        "https://example.com/ai-progress-2024-1",
        "https://example.com/embodied-ai",
        "https://example.com/generative-ai-apps",
        "https://example.com/quantum-computing-basics",
        "https://example.com/edge-ai"
    ]
    print("\n正在模拟创建 LlamaIndex 索引...")
    llama_index_service.create_or_update_index(sample_documents, sample_doc_ids)
    print("模拟索引创建完成。")

    # 模拟工具调用
    test_query = "2024年大型语言模型有哪些新进展？"
    print(f"\n正在通过 rag_tool 调用执行 RAG 查询: '{test_query}'...")
    try:
        # 传入字典作为工具的输入，StructuredTool 会将这些 kwargs 传递给 _run_rag_query
        results = rag_tool.invoke({"query": test_query, "top_k": 2})

        if results:
            print(f"RAG 工具返回了 {len(results)} 条结果: ")
            for i, res in enumerate(results):
                print(f"  结果 {i + 1}: ")
                print(f"    内容: {res.content[:100]}...")
                print(f"    来源: {res.source}")
        else:
            print("RAG 工具没有返回任何结果。")
    except Exception as e:
        print(f"调用 RAG 工具时发生错误: {e}")
