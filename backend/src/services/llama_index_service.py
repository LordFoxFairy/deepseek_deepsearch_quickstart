import logging
from typing import List, Optional, Dict, Any

from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import BaseNode, QueryBundle
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever

from backend.src.config.settings import settings
from backend.src.llms.openai_llm import get_chat_model
from backend.src.schemas.tool_models import SearchResult, RagResult

logger = logging.getLogger(__name__)


class SafeVectorIndexRetriever(VectorIndexRetriever):
    """一个自定义的、更安全的检索器，用于优雅地处理空查询结果。"""

    def _retrieve(self, query_bundle: QueryBundle) -> List[BaseNode]:
        try:
            return super()._retrieve(query_bundle)
        except AssertionError:
            logger.warning("SafeVectorIndexRetriever 检测到内部断言错误（查询结果为空），将安全地返回一个空列表。")
            return []


class LlamaIndexService:
    def __init__(self):
        logger.info("正在初始化 LlamaIndex 服务...")
        self.embed_model = DashScopeEmbeddings(
            model=settings.DASH_SCOPE_EMBEDDING_MODEL,
            dashscope_api_key=settings.DASH_SCOPE_API_KEY
        )
        self.llm = get_chat_model(
            model=settings.DEEPSEEK_CHAT_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        self.storage_context = StorageContext.from_defaults()
        self.index: VectorStoreIndex = VectorStoreIndex.from_documents([], storage_context=self.storage_context)
        logger.info("LlamaIndex 服务初始化完成。")

    def add_search_results_to_index(self, search_results: List[SearchResult],
                                    metadata: Optional[Dict[str, Any]] = None):
        if not search_results: return
        documents_to_add = []
        for res in search_results:
            content = f"标题: {res.title}\n摘要: {res.snippet}"
            if not res.url: continue
            doc_metadata = {"url": res.url, "title": res.title}
            if metadata: doc_metadata.update(metadata)
            documents_to_add.append(Document(text=content, metadata=doc_metadata))
        if not documents_to_add: return
        nodes = SentenceSplitter(chunk_size=512, chunk_overlap=20).get_nodes_from_documents(documents_to_add)
        self.index.insert_nodes(nodes)

    def _query_and_get_rag_results(self, query: str, filter_key: str, filter_values: List[str]) -> List[RagResult]:
        """
        (新增) 内部查询方法，返回结构化的 RagResult 列表。
        """
        logger.info(f"正在执行内部RAG查询: key='{filter_key}', values='{filter_values}'")
        try:
            retriever = SafeVectorIndexRetriever(index=self.index, filters=MetadataFilters(
                filters=[ExactMatchFilter(key=filter_key, value=val) for val in filter_values],
                condition="or"
            )) if filter_values else self.index.as_retriever()

            query_engine = RetrieverQueryEngine.from_args(retriever)
            response = query_engine.query(query)

            if not response.source_nodes:
                return []

            rag_results = []
            for node in response.source_nodes:
                rag_results.append(RagResult(
                    content=node.text,
                    source=node.metadata.get("url", "No URL found")
                ))
            return rag_results

        except Exception as e:
            logger.error(f"LlamaIndex 内部查询期间发生未知错误: {e}", exc_info=True)
            return []

    def query_index_with_metadata_filter(self, query: str, filter_key: str, filter_values: List[str]) -> str:
        """
        面向 Agent 的外部接口，调用内部查询方法，并返回格式化的字符串。
        """
        rag_results = self._query_and_get_rag_results(query, filter_key, filter_values)

        if not rag_results:
            return "根据提供的相关研究资料，未能找到关于此主题的特定信息。"

        # 将结构化结果格式化为对LLM友好的字符串
        context_parts = []
        for result in rag_results:
            context_parts.append(f"Source: {result.source}\nContent: {result.content}")

        return "\n\n---\n\n".join(context_parts)

    def get_document_by_source_url(self, url: str) -> Optional[BaseNode]:
        """通过源URL从文档存储中检索节点，用于获取引用标题。"""
        try:
            docstore = self.storage_context.docstore
            for doc_node in docstore.docs.values():
                if doc_node.metadata and doc_node.metadata.get("url") == url:
                    return doc_node
        except Exception as e:
            logger.error(f"在 get_document_by_source_url 中遍历文档时出错: {e}")
        return None


llama_index_service = LlamaIndexService()
