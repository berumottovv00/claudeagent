"""
Milvus 向量存储客户端
存储会话摘要的向量，支持语义相似度召回。
Milvus 不可用时自动降级，不影响主流程。

Collection 结构：
  ltm_summaries：
    id        — INT64，主键，自增
    user_id   — VARCHAR(64)
    summary   — VARCHAR(2048)，原始摘要文本
    embedding — FLOAT_VECTOR(dim)
"""
import logging
import os

logger = logging.getLogger(__name__)

MILVUS_HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "ltm_summaries")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))
TOP_K = int(os.environ.get("MILVUS_TOP_K", "3"))


class MilvusClient:
    def __init__(self):
        self._collection = None
        self._available = False
        try:
            from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility, IndexType
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            self._collection = self._get_or_create_collection(Collection, CollectionSchema, FieldSchema, DataType, utility, IndexType)
            self._available = True
        except Exception as e:
            logger.warning(f"Milvus 初始化失败，向量记忆降级忽略：{e}")

    def _get_or_create_collection(self, Collection, CollectionSchema, FieldSchema, DataType, utility, IndexType):
        if utility.has_collection(MILVUS_COLLECTION):
            col = Collection(MILVUS_COLLECTION)
            col.load()
            return col

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]
        schema = CollectionSchema(fields=fields, description="长期记忆会话摘要向量库")
        col = Collection(name=MILVUS_COLLECTION, schema=schema)
        col.create_index(
            field_name="embedding",
            index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}},
        )
        col.load()
        return col

    def insert(self, user_id: str, summary: str, embedding: list) -> None:
        """插入一条会话摘要向量，失败时静默降级。"""
        if not self._available:
            return
        try:
            self._collection.insert([[user_id], [summary], [embedding]])
        except Exception as e:
            logger.warning(f"Milvus 写入失败：{e}")

    def search(self, user_id: str, query_embedding: list) -> list[str]:
        """按语义召回该用户最相关的历史摘要，失败时返回空列表。"""
        if not self._available:
            return []
        try:
            results = self._collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"nprobe": 16}},
                limit=TOP_K,
                expr=f'user_id == "{user_id}"',
                output_fields=["summary"],
            )
            return [hit.entity.get("summary") for hit in results[0]]
        except Exception as e:
            logger.warning(f"Milvus 检索失败：{e}")
            return []


milvus_client = MilvusClient()