from typing import List

from googlesearch import search, SearchResult as GoogleSearchResult

from backend.src.schemas.tool_models import SearchResult


class SearchAPIService:
    """
    封装了与外部搜索API的交互逻辑。
    当前实现使用了 `googlesearch-python` 库。
    """

    def __init__(self):
        """
        初始化搜索服务。
        """
        pass  # 当前实现不需要特定的初始化参数

    @staticmethod
    def perform_search(query: str, num_results: int = 20) -> List[SearchResult]:
        """
        执行网络搜索并返回结构化的搜索结果列表。

        参数:
            query (str): 需要搜索的查询字符串。
            num_results (int): 希望返回的搜索结果数量。

        返回:
            List[SearchResult]: 结构化的搜索结果列表。
        """
        print(f"正在执行搜索查询: {query}，请求 {num_results} 条结果 (使用 googlesearch 库)...")
        results: List[SearchResult] = []
        try:
            # 调用googlesearch库的search函数，设置advanced=True以获取更详细的SearchResult对象。
            # search函数返回一个生成器。
            search_generator = search(term=query, num_results=num_results, advanced=True, timeout=30)

            for item in search_generator:
                # 确保返回的item是googlesearch.SearchResult的实例
                if isinstance(item, GoogleSearchResult):
                    results.append(
                        SearchResult(
                            title=item.title if item.title else "无标题",
                            url=item.url if item.url else "#",
                            snippet=item.description if item.description else "无摘要"
                        )
                    )
                else:
                    print(f"警告: googlesearch 返回了非预期的类型: {type(item)}，跳过此结果。")

            print(f"搜索完成，成功解析并返回 {len(results)} 条结果。")
            return results
        except Exception as e:
            print(f"执行搜索时发生错误: {e}")
            print("请确保 googlesearch 库已正确安装，并且网络连接正常。")
            return []


if __name__ == "__main__":
    # 此代码块仅用于测试SearchAPIService的功能
    search_service = SearchAPIService()

    # 模拟搜索查询
    test_query = "2024年人工智能最新发展"
    results = search_service.perform_search(test_query, num_results=3)

    if results:
        print(f"\n搜索查询 '{test_query}' 的结果: ")
        for i, res in enumerate(results):
            print(f"  结果 {i + 1}: ")
            print(f"  标题: {res.title}")
            print(f"  URL: {res.url}")
            print(f"  摘要: {res.snippet[:100]}...")
    else:
        print(f"\n没有找到 '{test_query}' 的搜索结果或发生错误。")
        print("请检查上述日志输出以获取更多调试信息。")
