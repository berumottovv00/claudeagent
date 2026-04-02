"""
Milvus 向量存储客户端
存储会话摘要的向量，支持语义相似度召回。
Milvus 不可用时自动降级，不影响主流程。

Collection 结构：
  ltm_summaries：
    id         — INT64，主键，自增
    user_id    — VARCHAR(64)
    summary    — VARCHAR(2048)，原始摘要文本
    embedding  — FLOAT_VECTOR(dim)
    created_at — INT64，Unix 时间戳，用于 TTL 过期清理
"""
import logging
import os
import time
from typing import List, Tuple

logger = logging.getLogger(__name__)

MILVUS_HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "ltm_summaries")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))
TOP_K = int(os.environ.get("MILVUS_TOP_K", "3"))
DEDUP_THRESHOLD = float(os.environ.get("MILVUS_DEDUP_THRESHOLD", "0.95"))


class MilvusClient:
    def __init__(self):
        self._collection = None
        self._available = False
        try:
            from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            self._collection = self._get_or_create_collection(
                Collection, CollectionSchema, FieldSchema, DataType, utility
            )
            self._available = True
        except Exception as e:
            logger.warning(f"Milvus 初始化失败，向量记忆降级忽略：{e}")

    def _get_or_create_collection(self, Collection, CollectionSchema, FieldSchema, DataType, utility):
        # 删除旧 collection（重建以支持新 schema）
        if utility.has_collection(MILVUS_COLLECTION):
            utility.drop_collection(MILVUS_COLLECTION)
            logger.info(f"已删除旧 Milvus collection：{MILVUS_COLLECTION}")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields, description="长期记忆会话摘要向量库")
        col = Collection(name=MILVUS_COLLECTION, schema=schema)
        col.create_index(
            field_name="embedding",
            index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}},
        )
        col.load()
        logger.info(f"已创建新 Milvus collection：{MILVUS_COLLECTION}")
        return col

    def insert(self, user_id: str, summary: str, embedding: list) -> List[int]:
        """插入一条摘要向量，返回 Milvus 分配的主键 ID 列表。失败时返回空列表。"""
        if not self._available:
            return []
        try:
            result = self._collection.insert(
                [[user_id], [summary], [embedding], [int(time.time())]]
            )
            return list(result.primary_keys)
        except Exception as e:
            logger.warning(f"Milvus 写入失败：{e}")
            return []

    def delete(self, ids: List[int]) -> None:
        """按主键 ID 删除向量，失败时静默降级。"""
        if not self._available or not ids:
            return
        try:
            expr = f"id in [{', '.join(str(i) for i in ids)}]"
            self._collection.delete(expr)
        except Exception as e:
            logger.warning(f"Milvus 删除失败：{e}")

    def delete_expired(self, user_id: str, cutoff_timestamp: int) -> None:
        """删除指定用户在截止时间戳之前创建的向量（用于 TTL 清理）。"""
        if not self._available:
            return
        try:
            expr = f'user_id == "{user_id}" and created_at < {cutoff_timestamp}'
            self._collection.delete(expr)
        except Exception as e:
            logger.warning(f"Milvus TTL 清理失败：{e}")

    def find_similar(self, user_id: str, embedding: list, threshold: float = DEDUP_THRESHOLD) -> Tuple[int, str] | None:
        """查找相似度超过阈值的已有向量，返回 (id, summary)，不存在则返回 None。"""
        if not self._available:
            return None
        try:
            results = self._collection.search(
                data=[embedding],
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"nprobe": 16}},
                limit=1,
                expr=f'user_id == "{user_id}"',
                output_fields=["summary"],
            )
            if results and results[0]:
                hit = results[0][0]
                if hit.score >= threshold:
                    return hit.id, hit.entity.get("summary")
        except Exception as e:
            logger.warning(f"Milvus 相似度查找失败：{e}")
        return None

    def find_related(self, user_id: str, embedding: list, top_k: int = 5, threshold: float = 0.6) -> List[Tuple[int, str]]:
        """查找语义相关（低于去重阈值）的历史摘要，用于偏好冲突检测。"""
        if not self._available:
            return []
        try:
            results = self._collection.search(
                data=[embedding],
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"nprobe": 16}},
                limit=top_k,
                expr=f'user_id == "{user_id}"',
                output_fields=["summary"],
            )
            return [
                (hit.id, hit.entity.get("summary"))
                for hit in results[0]
                if hit.score >= threshold
            ]
        except Exception as e:
            logger.warning(f"Milvus 相关摘要查找失败：{e}")
            return []

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
