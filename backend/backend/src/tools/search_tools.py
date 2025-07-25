from langchain_core.tools import StructuredTool
from backend.src.services.search_api_service import SearchAPIService
from backend.src.schemas.tool_models import SearchToolInput, SearchResult
from typing import List, Any


def _run_search(**kwargs: Any) -> List[SearchResult]:
    """
    执行网络搜索，并返回结构化的搜索结果。
    此函数是一个包装器，用于将SearchAPIService的搜索结果适配为Langchain工具的输出格式。

    参数:
        **kwargs: 包含查询字符串和待检索结果数量的关键字参数。
                  这些参数将被用于创建SearchToolInput实例。

    返回:
        List[SearchResult]: 包含搜索结果标题、URL和摘要的列表。
    """
    try:
        # 从kwargs创建SearchToolInput实例
        tool_input = SearchToolInput(**kwargs)
    except Exception as e:
        print(f"创建 SearchToolInput 失败: {e}. 传入的参数: {kwargs}")
        raise ValueError(f"无效的工具输入: {e}")

    # 从Pydantic输入模型中解构参数
    query = tool_input.query
    num_results = tool_input.num_results

    # 调用SearchAPIService执行搜索
    return SearchAPIService.perform_search(query=query, num_results=num_results)


# 定义Langchain搜索工具
# 使用StructuredTool以确保输入参数的类型安全
search_tool = StructuredTool(
    name="search_tool",
    description="用于执行网络搜索以获取最新信息的工具。输入是一个搜索查询字符串和可选的返回结果数量。",
    func=_run_search,
    args_schema=SearchToolInput
)
