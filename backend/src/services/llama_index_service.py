import logging  # 引入日志模块
from typing import List, Optional

from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import SimpleVectorStore

from backend.src.config.settings import settings
from backend.src.llms.openai_llm import get_chat_model
from backend.src.schemas.tool_models import RagResult, SearchResult

# 配置日志
logger = logging.getLogger(__name__)


class LlamaIndexService:
    """
    封装了LlamaIndex的索引创建、管理和检索逻辑。
    """

    def __init__(self):
        """
        初始化LlamaIndex服务。
        """
        logger.info("正在初始化 LlamaIndex 服务...")
        # 配置嵌入模型
        self.embed_model = DashScopeEmbeddings(
            model=settings.DASH_SCOPE_EMBEDDING_MODEL,
            dashscope_api_key=settings.DASH_SCOPE_API_KEY
        )

        # 配置LlamaIndex使用的LLM
        self.llm = get_chat_model(
            model=settings.DEEPSEEK_CHAT_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )

        self.vector_store = SimpleVectorStore()
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index: Optional[VectorStoreIndex] = None
        self._initialize_index()

    def _initialize_index(self):
        """
        初始化或加载向量索引。
        """
        logger.info("确保 LlamaIndex 索引已初始化...")
        if self.index is None:
            self.index = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
                embed_model=self.embed_model,
            )
        logger.info("LlamaIndex 服务初始化完成。")

    def create_or_update_index(self, documents: List[str], doc_ids: Optional[List[str]] = None):
        """
        从文本内容创建或更新LlamaIndex索引。
        """
        logger.info(f"正在创建/更新 LlamaIndex 索引，共 {len(documents)} 篇文档...")
        llama_documents = []
        for i, doc_content in enumerate(documents):
            metadata = {"doc_id": doc_ids[i]} if doc_ids else {"doc_id": f"doc_{i}"}
            llama_documents.append(Document(text=doc_content, metadata=metadata))

        parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(llama_documents)

        if self.index is None:
            self._initialize_index()  # 确保索引已初始化

        self.index.insert_nodes(nodes)
        logger.info(f"LlamaIndex 索引创建/更新完成，共 {len(nodes)} 个节点。")

    def add_search_results_to_index(self, search_results: List[SearchResult]):
        """
        将网页搜索结果添加到LlamaIndex索引中。
        """
        if not search_results:
            logger.warning("没有搜索结果可添加到 LlamaIndex。")
            return

        logger.info(f"正在将 {len(search_results)} 条搜索结果添加到 LlamaIndex 索引...")
        documents_to_add = []
        for res in search_results:
            content = f"标题: {res.title}\n摘要: {res.snippet}"
            documents_to_add.append(Document(text=content, metadata={"url": res.url, "title": res.title}))

        parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(documents_to_add)

        if self.index is None:
            self._initialize_index()

        self.index.insert_nodes(nodes)
        logger.info(f"已将 {len(nodes)} 个节点从搜索结果添加到 LlamaIndex。")

    def retrieve(self, query: str, top_k: int = 3) -> List[RagResult]:
        """
        从LlamaIndex索引中检索与查询最相关的文档片段。
        """
        if self.index is None:
            logger.warning("LlamaIndex 尚未初始化，无法执行检索。")
            return []

        logger.info(f"正在从 LlamaIndex 检索与查询 '{query}' 相关的 top {top_k} 条结果...")
        retriever = self.index.as_retriever(similarity_top_k=top_k)

        try:
            nodes_with_score: List[NodeWithScore] = retriever.retrieve(query)
        except AssertionError as e:
            # 捕获 LlamaIndex 在找不到节点时可能抛出的内部断言错误
            logger.warning(f"LlamaIndex retriever 触发了一个断言错误: {e}。这通常意味着没有找到任何结果。将返回空列表。")
            nodes_with_score = []

        results: List[RagResult] = []
        for node in nodes_with_score:
            source_url = node.metadata.get("url", node.metadata.get("doc_id", "未知来源"))
            results.append(
                RagResult(
                    content=node.text,
                    source=source_url
                )
            )
        logger.info(f"LlamaIndex 检索完成，返回 {len(results)} 条结果。")
        return results

    def query_index(self, query: str) -> str:
        """
        直接向LlamaIndex索引发起查询，并由其查询引擎生成答案。
        """
        if self.index is None:
            logger.warning("LlamaIndex 尚未初始化，无法执行查询。")
            return "LlamaIndex 服务未准备好。"

        logger.info(f"正在向 LlamaIndex 查询引擎发起查询: {query}...")
        query_engine = self.index.as_query_engine(llm=self.llm)
        response = query_engine.query(query)
        logger.info("LlamaIndex 查询引擎响应完成。")
        return str(response)


llama_index_service = LlamaIndexService()
