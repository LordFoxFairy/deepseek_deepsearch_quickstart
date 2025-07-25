from typing import Optional

from pydantic import BaseModel, Field


# 搜索工具的输入模型
class SearchToolInput(BaseModel):
    """
    搜索工具的输入参数模型。
    """
    query: str = Field(..., description="用于执行网络搜索的查询字符串。")
    num_results: int = Field(5, description="要返回的搜索结果数量。", ge=1, le=10)


# 搜索结果模型
class SearchResult(BaseModel):
    """
    单个搜索结果的结构。
    """
    title: str = Field(..., description="搜索结果的标题。")
    url: str = Field(..., description="搜索结果的 URL。")
    snippet: str = Field(..., description="搜索结果的摘要。")


# RAG（检索增强生成）工具的输入模型
class RagToolInput(BaseModel):
    """
    RAG 工具的输入参数模型。
    """
    query: str = Field(..., description="用于从知识库中检索相关信息的查询。")
    top_k: int = Field(3, description="要检索的相似文档数量。", ge=1)


# RAG 检索结果模型
class RagResult(BaseModel):
    """
    单个 RAG 检索结果的结构。
    """
    content: str = Field(..., description="检索到的文档内容片段。")
    source: Optional[str] = Field(None, description="检索结果的来源（例如，文档ID或URL）。")
