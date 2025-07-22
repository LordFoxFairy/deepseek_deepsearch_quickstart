from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from langchain_community.embeddings import DashScopeEmbeddings  # 导入 DashScopeEmbeddings
from llama_index.llms.openai import OpenAI  # LlamaIndex 的 OpenAI LLM 模块可以配置 api_base
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import SimpleVectorStore  # 仅用于演示，实际项目中应使用持久化向量存储
from backend.src.config.settings import settings
from backend.src.schemas.tool_models import RagResult, SearchResult  # 导入 SearchResult


class LlamaIndexService:
    """
    封装 LlamaIndex 的索引创建、管理和检索逻辑。
    """

    def __init__(self):
        """
        初始化 LlamaIndex 服务。
        设置 DashScope 嵌入模型和 DeepSeek LLM。
        注意：SimpleVectorStore 仅用于演示，数据不会持久化。
        在实际应用中，应配置 ChromaDB, Pinecone, Weaviate 等持久化向量存储。
        """
        # 配置 DashScope 嵌入模型
        self.embed_model = DashScopeEmbeddings(
            model=settings.DASH_SCOPE_EMBEDDING_MODEL,
            dashscope_api_key=settings.DASH_SCOPE_API_KEY
        )

        # 配置 LlamaIndex 的 LLM，使用 DeepSeek
        self.llm = OpenAI(
            model=settings.DEEPSEEK_CHAT_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            api_base=settings.DEEPSEEK_BASE_URL
        )

        # 初始化一个简单的内存向量存储，仅用于演示
        self.vector_store = SimpleVectorStore()
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # 初始化索引，如果需要持久化，这里需要加载现有索引
        self.index: Optional[VectorStoreIndex] = None
        self._initialize_index()

    def _initialize_index(self):
        """
        初始化或加载向量索引。
        在实际应用中，这里应该包含加载持久化索引的逻辑。
        """
        print("正在初始化 LlamaIndex 服务...")
        # 确保在初始化时创建一个 VectorStoreIndex 实例
        # 即使初始为空，也保证 self.index 不为 None
        if self.index is None:
            self.index = VectorStoreIndex.from_documents(
                [],  # 从空文档列表创建索引
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                llm=self.llm  # 可以指定 LLM 用于某些索引操作
            )
        print("LlamaIndex 服务初始化完成。")

    def create_or_update_index(self, documents: List[str], doc_ids: Optional[List[str]] = None):
        """
        从文本内容创建或更新 LlamaIndex 索引。
        这将把文本分割成节点并嵌入，然后存储到向量存储中。

        Args:
            documents (List[str]): 要索引的文本内容列表。
            doc_ids (Optional[List[str]]): 对应文档的唯一ID列表。
        """
        print(f"正在创建/更新 LlamaIndex 索引，共 {len(documents)} 篇文档...")
        llama_documents = []
        for i, doc_content in enumerate(documents):
            # 可以为每个文档添加元数据，例如 source URL
            metadata = {"doc_id": doc_ids[i]} if doc_ids else {"doc_id": f"doc_{i}"}
            llama_documents.append(Document(text=doc_content, metadata=metadata))

        # 使用 SentenceSplitter 将文档分割成更小的节点
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(llama_documents)

        # 如果索引不存在，则创建新索引；否则，添加新节点
        if self.index is None:
            # 这种情况在 _initialize_index 修复后应该不会发生，但作为安全措施保留
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                llm=self.llm  # 可以指定 LLM 用于某些索引操作
            )
        else:
            # 对于 SimpleVectorStore，直接添加节点即可
            # 对于其他向量存储，可能需要调用特定的添加方法
            self.index.insert_nodes(nodes)

        print(f"LlamaIndex 索引创建/更新完成，共 {len(nodes)} 个节点。")

    def add_search_results_to_index(self, search_results: List[SearchResult]):
        """
        将网页搜索结果（SearchResult 对象列表）添加到 LlamaIndex 索引中。
        这使得后续的 RAG 查询可以针对这些动态获取的信息进行。

        Args:
            search_results (List[SearchResult]): 从搜索工具获取的结构化搜索结果列表。
        """
        if not search_results:
            print("没有搜索结果可添加到 LlamaIndex。")
            return

        print(f"正在将 {len(search_results)} 条搜索结果添加到 LlamaIndex 索引...")
        documents_to_add = []
        for res in search_results:
            # 将搜索结果的标题和摘要作为文档内容，URL 作为元数据
            content = f"标题: {res.title}\n摘要: {res.snippet}"
            documents_to_add.append(Document(text=content, metadata={"url": res.url, "title": res.title}))

        # 使用 SentenceSplitter 将文档分割成更小的节点
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(documents_to_add)

        if self.index is None:
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                llm=self.llm
            )
        else:
            self.index.insert_nodes(nodes)

        print(f"已将 {len(nodes)} 个节点从搜索结果添加到 LlamaIndex。")

    def retrieve(self, query: str, top_k: int = 3) -> List[RagResult]:
        """
        从 LlamaIndex 索引中检索与查询最相关的文档片段。

        Args:
            query (str): 用于检索的查询字符串。
            top_k (int): 要检索的最相关文档片段的数量。

        Returns:
            List[RagResult]: 检索到的文档内容和来源列表。
        """
        if self.index is None:
            print("LlamaIndex 尚未初始化，无法执行检索。")
            return []

        print(f"正在从 LlamaIndex 检索与查询 '{query}' 相关的 top {top_k} 条结果...")
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes_with_score: List[NodeWithScore] = retriever.retrieve(query)

        results: List[RagResult] = []
        for node in nodes_with_score:
            # 来源现在可以从元数据中获取，无论是初始索引还是搜索结果索引
            source_url = node.metadata.get("url", node.metadata.get("doc_id", "未知来源"))
            results.append(
                RagResult(
                    content=node.text,
                    source=source_url
                )
            )
        print(f"LlamaIndex 检索完成，返回 {len(results)} 条结果。")
        return results

    def query_index(self, query: str) -> str:
        """
        直接向 LlamaIndex 索引发起查询，并由 LlamaIndex 的查询引擎生成答案。
        这通常会结合检索和 LLM 综合。

        Args:
            query (str): 用户查询。

        Returns:
            str: LlamaIndex 查询引擎生成的答案。
        """
        if self.index is None:
            print("LlamaIndex 尚未初始化，无法执行查询。")
            return "LlamaIndex 服务未准备好。"

        print(f"正在向 LlamaIndex 查询引擎发起查询: {query}...")
        query_engine = self.index.as_query_engine(llm=self.llm)  # 使用配置的 LLM
        response = query_engine.query(query)
        print("LlamaIndex 查询引擎响应完成。")
        return str(response)


# 在模块级别初始化 LlamaIndexService 实例，供其他模块（如 rag_tool）使用
llama_index_service = LlamaIndexService()

if __name__ == "__main__":
    # 此块用于测试 LlamaIndexService
    print("正在测试 llama_index_service.py...")

    # 模拟一些搜索结果来测试 add_search_results_to_index
    mock_search_results = [
        SearchResult(
            title="2024年AI趋势报告",
            url="https://example.com/ai-report-2024",
            snippet="这份报告详细介绍了2024年人工智能的十大关键趋势，包括生成式AI、具身智能和边缘计算。"
        ),
        SearchResult(
            title="LLM最新进展",
            url="https://example.com/llm-updates",
            snippet="大型语言模型在2024年取得了显著突破，尤其是在多模态能力和长上下文处理方面。"
        )
    ]

    print("\n正在将模拟搜索结果添加到 LlamaIndex...")
    llama_index_service.add_search_results_to_index(mock_search_results)
    print("模拟搜索结果添加完成。")

    # 尝试检索
    test_query = "2024年LLM有哪些新进展？"
    retrieved_results = llama_index_service.retrieve(test_query, top_k=2)

    if retrieved_results:
        print(f"\n检索查询 '{test_query}' 的结果:")
        for i, res in enumerate(retrieved_results):
            print(f"  结果 {i + 1}:")
            print(f"    内容: {res.content[:100]}...")
            print(f"    来源: {res.source}")
    else:
        print(f"\n没有找到 '{test_query}' 的检索结果或发生错误。")
