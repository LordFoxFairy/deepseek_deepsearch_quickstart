from typing import List, Optional

from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from backend.src.config.settings import settings
from backend.src.schemas.tool_models import RagResult


class LlamaIndexService:
    """
    封装 LlamaIndex 的索引创建、管理和检索逻辑。
    """

    def __init__(self):
        """
        初始化 LlamaIndex 服务。
        设置 OpenAI 嵌入模型和 LLM。
        注意：SimpleVectorStore 仅用于演示，数据不会持久化。
        在实际应用中，应配置 ChromaDB, Pinecone, Weaviate 等持久化向量存储。
        """
        # 配置 OpenAI 嵌入模型
        # 注意：这里使用 llama_index.embeddings.openai.OpenAIEmbedding
        # 它会使用环境变量或传入的 api_key 和 api_base
        self.embed_model = OpenAIEmbedding(
            model=settings.OPENAI_EMBEDDING_MODEL,  # 确保 settings 中有此配置
            api_key=settings.OPENAI_API_KEY,
            api_base=settings.OPENAI_BASE_URL
        )

        # 配置 LlamaIndex 的 LLM，用于某些高级检索或查询转换（如果需要）
        self.llm = OpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            api_key=settings.OPENAI_API_KEY,
            api_base=settings.OPENAI_BASE_URL
        )

        # 初始化一个简单的内存向量存储，仅用于演示
        self.vector_store = SimpleVectorStore()
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # 初始化索引，如果需要持久化，这里需要加载现有索引
        self.index: Optional[VectorStoreIndex] = None
        self._initialize_index()

    @staticmethod
    def _initialize_index():
        """
        初始化或加载向量索引。
        在实际应用中，这里应该包含加载持久化索引的逻辑。
        """
        print("正在初始化 LlamaIndex 服务...")
        # 暂时创建一个空索引，或者可以从预定义文档加载
        # 如果是首次运行，可以构建索引
        # self.index = VectorStoreIndex([], storage_context=self.storage_context, embed_model=self.embed_model)
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
            source_url = node.metadata.get("doc_id", "未知来源")  # 假设 doc_id 存储了 URL
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
