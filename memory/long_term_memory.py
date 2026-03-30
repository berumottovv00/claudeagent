"""
长期记忆管理
按 user_id 存储到 Redis，跨会话持久化用户偏好摘要和历史会话摘要。
同时将摘要向量写入 Milvus，支持语义召回。
Redis 和 Milvus 不可用时自动降级，不影响主流程。

Redis key 结构：
  ltm:{user_id}:preferences  →  String，用户偏好摘要（每次会话后覆盖更新）
  ltm:{user_id}:summaries    →  List，历史会话摘要（LPUSH，保留最近 N 条）
"""
import json
import logging
import os
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .milvus_client import milvus_client
from .redis_client import RedisClient

logger = logging.getLogger(__name__)

LTM_TTL_SECONDS = int(os.environ.get("LTM_TTL_DAYS", "30")) * 86400
MAX_SUMMARIES = int(os.environ.get("LTM_MAX_SUMMARIES", "20"))

_SUMMARIZE_PROMPT = """请根据以下对话记录，用简洁的中文完成两件事：
1. 提取用户的偏好、习惯、关注点（如常用车型、出行习惯、关注问题等），用于个性化服务
2. 总结本次会话的核心内容（1-3 句话）

对话记录：
{conversation}

请严格按以下 JSON 格式输出，不要有多余内容：
{{"preferences": "...", "summary": "..."}}"""


class LongTermMemoryManager:
    def __init__(self):
        self._redis = RedisClient()
        self._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def _pref_key(self, user_id: str) -> str:
        return f"ltm:{user_id}:preferences"

    def _sum_key(self, user_id: str) -> str:
        return f"ltm:{user_id}:summaries"

    def get_context(self, user_id: str, query: str = "") -> str:
        """拉取用户长期记忆，返回可注入 system prompt 的文本。Redis 不可用时返回空字符串。"""
        try:
            preferences = self._redis.get(self._pref_key(user_id)) or ""
            summaries = self._redis.lrange(self._sum_key(user_id), 0, 4)  # 最近 5 条
        except Exception as e:
            logger.warning(f"长期记忆读取失败，降级忽略：{e}")
            return ""

        # 语义召回：用当前问题从 Milvus 检索相关历史摘要
        semantic_summaries = []
        if query:
            try:
                query_embedding = self._embeddings.embed_query(query)
                semantic_summaries = milvus_client.search(user_id, query_embedding)
            except Exception as e:
                logger.warning(f"语义记忆召回失败，降级忽略：{e}")

        semantic_summaries = [s for s in semantic_summaries if s not in summaries]

        parts = []
        if preferences:
            parts.append(f"用户偏好：{preferences}")
        if summaries:
            parts.append("最近会话摘要：\n" + "\n".join(f"- {s}" for s in summaries))
        if semantic_summaries:
            parts.append("语义相关历史摘要：\n" + "\n".join(f"- {s}" for s in semantic_summaries))
        return "\n".join(parts)

    def save_session(self, user_id: str, messages: List[BaseMessage], llm: ChatOpenAI) -> None:
        """会话结束时，用 LLM 压缩摘要后写入 Redis 和 Milvus。失败时静默降级。"""
        if not messages:
            return
        try:
            conversation = "\n".join(
                f"{'用户' if isinstance(m, HumanMessage) else 'AI'}：{m.content}"
                for m in messages
            )
            response = llm.invoke(_SUMMARIZE_PROMPT.format(conversation=conversation))
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content.strip())

            pipe = self._redis.pipeline()
            if result.get("preferences"):
                pipe.set(self._pref_key(user_id), result["preferences"], ex=LTM_TTL_SECONDS)
            if result.get("summary"):
                sum_key = self._sum_key(user_id)
                pipe.lpush(sum_key, result["summary"])
                pipe.ltrim(sum_key, 0, MAX_SUMMARIES - 1)
                pipe.expire(sum_key, LTM_TTL_SECONDS)
            pipe.execute()

            # 写入 Milvus 向量
            if result.get("summary"):
                embedding = self._embeddings.embed_query(result["summary"])
                milvus_client.insert(user_id, result["summary"], embedding)

            logger.info(f"长期记忆已保存：user_id={user_id}")
        except Exception as e:
            logger.warning(f"长期记忆写入失败，降级忽略：{e}")

    def clear(self, user_id: str) -> None:
        """清除指定用户的全部长期记忆。"""
        try:
            self._redis.delete(self._pref_key(user_id), self._sum_key(user_id))
        except Exception as e:
            logger.warning(f"长期记忆清除失败：{e}")


long_term_memory_manager = LongTermMemoryManager()