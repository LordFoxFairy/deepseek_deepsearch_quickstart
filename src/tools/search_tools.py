from langchain_core.tools import StructuredTool
from backend.src.schemas.tool_models import SearchToolInput, SearchResult
from typing import List, Any  # 导入 Any 用于 kwargs 类型提示

from backend.src.services.search_api_service import SearchAPIService

# 初始化搜索服务实例
search_service = SearchAPIService()


def _run_search(**kwargs: Any) -> List[SearchResult]:
    """
    执行网络搜索并返回结构化的搜索结果。
    这是一个包装函数，用于将 SearchAPIService 的结果适配为工具的输出。

    Args:
        **kwargs: 包含查询字符串和要返回结果数量的关键字参数。
                  这些参数将被用于创建 SearchToolInput 实例。

    Returns:
        List[SearchResult]: 结构化的搜索结果列表。
    """
    try:
        # 从 kwargs 创建 SearchToolInput 实例
        tool_input = SearchToolInput(**kwargs)
    except Exception as e:
        print(f"创建 SearchToolInput 失败: {e}. 传入的参数: {kwargs}")
        # 可以选择抛出异常或返回空列表
        raise ValueError(f"无效的工具输入: {e}")

    query = tool_input.query
    num_results = tool_input.num_results
    return search_service.perform_search(query=query, num_results=num_results)


# 定义 Langchain 搜索工具
# 使用 StructuredTool 来确保 Pydantic 输入模型被正确处理。
search_tool = StructuredTool(
    name="search_tool",
    description="用于执行网络搜索并获取最新信息的工具。输入是一个搜索查询字符串和可选的返回结果数量。",
    func=_run_search,
    args_schema=SearchToolInput  # 使用 Pydantic 模型作为工具的输入 Schema
)

if __name__ == "__main__":
    # 此块用于测试 search_tools.py
    print("正在测试 search_tools.py...")

    # 模拟工具调用
    test_query = "2024年人工智能领域最重要的突破"
    print(f"\n正在通过 search_tool 调用执行搜索: '{test_query}'...")
    try:
        # 传入字典作为工具的输入，StructuredTool 会将这些 kwargs 传递给 _run_search
        results = search_tool.invoke({"query": test_query, "num_results": 2})
        if results:
            print(f"搜索工具返回了 {len(results)} 条结果:")
            for i, res in enumerate(results):
                print(f"  结果 {i + 1}:")
                print(f"    标题: {res.title}")
                print(f"    URL: {res.url}")
                print(f"    摘要: {res.snippet[:100]}...")
        else:
            print("搜索工具没有返回任何结果。")
    except Exception as e:
        print(f"调用搜索工具时发生错误: {e}")
