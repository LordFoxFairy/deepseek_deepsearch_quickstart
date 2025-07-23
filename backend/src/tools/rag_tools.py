from langchain_core.tools import StructuredTool
from backend.src.services.llama_index_service import llama_index_service
from backend.src.schemas.tool_models import RagToolInput, RagResult
from typing import List, Any


def _run_rag_query(**kwargs: Any) -> List[RagResult]:
    """
    执行RAG（检索增强生成）查询，从知识库中检索相关信息。
    此函数是一个包装器，用于将LlamaIndexService的检索结果适配为Langchain工具的输出格式。

    参数:
        **kwargs: 包含查询字符串和待检索文档数量的关键字参数。
                  这些参数将被用于创建RagToolInput实例。

    返回:
        List[RagResult]: 包含检索到的文档内容和来源的列表。
    """
    try:
        # 从kwargs创建RagToolInput实例
        tool_input = RagToolInput(**kwargs)
    except Exception as e:
        print(f"创建 RagToolInput 失败: {e}. 传入的参数: {kwargs}")
        raise ValueError(f"无效的工具输入: {e}")

    # 从Pydantic输入模型中解构参数
    query = tool_input.query
    top_k = tool_input.top_k

    return llama_index_service.retrieve(query=query, top_k=top_k)


# 定义Langchain RAG工具
# 使用StructuredTool以确保Pydantic输入模型被正确处理
rag_tool = StructuredTool(
    name="rag_tool",
    description="用于从内部知识库或文档中检索相关信息的工具。输入是一个查询字符串和可选的返回文档数量。",
    func=_run_rag_query,
    args_schema=RagToolInput
)
