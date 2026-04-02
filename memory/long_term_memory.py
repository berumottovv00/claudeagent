"""
长期记忆管理 - 支持偏好冲突消解与基于重要性的遗忘策略

偏好冲突消解：
  将偏好从单一字符串升级为带置信度的结构化事实列表，
  每次会话结束时通过 LLM 合并新旧偏好，自动识别矛盾并更新置信度。
  与新信息矛盾的偏好置信度 -1，降至 ≤ 2 时自动遗忘。

遗忘策略：
  摘要使用 Redis Sorted Set 存储（score = 写入时间戳）。
  记录每条摘要的语义召回次数，容量超限时按综合重要性淘汰：
    score = 0.7 × 时效性衰减 + 0.3 × 访问频率（归一化）
  淘汰时同步删除 Milvus 中对应的向量，保持数据一致。

近重复去重：
  写入 Milvus 前检查相似度，相似度 > 阈值时替换旧向量而非累积。

Redis key 结构：
  ltm:{user_id}:preferences  → String(JSON)，结构化偏好事实列表
  ltm:{user_id}:summaries    → Sorted Set，score=时间戳，member=摘要文本
  ltm:{user_id}:sum_access   → Hash，md5(摘要) → 访问次数
  ltm:{user_id}:sum_ids      → Hash，md5(摘要) → Milvus 主键 ID
"""
import hashlib
import json
import logging
import math
import os
import time
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .milvus_client import milvus_client
from .redis_client import RedisClient

logger = logging.getLogger(__name__)

LTM_TTL_SECONDS = int(os.environ.get("LTM_TTL_DAYS", "30")) * 86400
MAX_SUMMARIES = int(os.environ.get("LTM_MAX_SUMMARIES", "20"))
DECAY_RATE = float(os.environ.get("LTM_DECAY_RATE", "0.05"))  # 每天衰减率

_SUMMARIZE_PROMPT = """请根据以下对话记录，用简洁的中文完成两件事：
1. 提取用户的偏好、习惯、关注点（如常用车型、出行习惯、关注问题等），用于个性化服务
2. 总结本次会话的核心内容（1-3 句话）

对话记录：
{conversation}

请严格按以下 JSON 格式输出，不要有多余内容：
{{"preferences": "...", "summary": "..."}}"""

_CONFLICT_SUMMARY_PROMPT = """你是记忆冲突检测专家。判断新摘要与历史摘要之间是否存在用户偏好矛盾。

【新摘要】：
{new_summary}

【历史摘要列表】（编号从 0 开始）：
{old_summaries}

判断标准：若某条历史摘要描述的用户偏好与新摘要中的偏好明显矛盾（如"喜欢辣" vs "不能吃辣"），则认为冲突。
仅考虑明确的偏好矛盾，不要将不同场景的描述误判为冲突。

请严格按以下 JSON 格式输出，conflict_indices 为存在冲突的历史摘要编号列表（无冲突则为空列表）：
{{"conflict_indices": [0, 2]}}"""

_MERGE_PREFERENCES_PROMPT = """你是用户偏好管理专家。请将新偏好合并到现有偏好列表中。

【现有偏好列表】（置信度 1-10，越高越可靠）：
{existing_facts}

【本次会话新提取的偏好描述】：
{new_preferences}

合并规则：
1. 与现有偏好一致 → 该条置信度 +1（最高 10）
2. 与现有偏好矛盾 → 被矛盾的那条置信度 -1（不立即删除，允许多次矛盾后渐进遗忘）
3. 全新偏好信息 → 以置信度 5 加入
4. 置信度 ≤ 2 → 从列表删除（遗忘）

请严格按以下 JSON 格式输出，不含多余内容：
{{"facts": [{{"content": "...", "confidence": 7}}, ...]}}"""


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


def _importance_score(timestamp: float, access_count: int, now: float) -> float:
    """综合重要性分数 = 0.7 × 时效性 + 0.3 × 访问热度（归一化）"""
    days_old = (now - timestamp) / 86400
    recency = math.exp(-DECAY_RATE * days_old)
    access_boost = min(math.log1p(access_count) / math.log1p(10), 1.0)
    return 0.7 * recency + 0.3 * access_boost


class LongTermMemoryManager:
    def __init__(self):
        self._redis = RedisClient()
        self._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ── key helpers ──

    def _pref_key(self, uid: str) -> str:
        return f"ltm:{uid}:preferences"

    def _sum_key(self, uid: str) -> str:
        return f"ltm:{uid}:summaries"

    def _access_key(self, uid: str) -> str:
        return f"ltm:{uid}:sum_access"

    def _ids_key(self, uid: str) -> str:
        return f"ltm:{uid}:sum_ids"

    # ── 读取 ──

    def get_context(self, user_id: str, query: str = "") -> str:
        """拉取用户长期记忆，返回可注入 system prompt 的文本。"""
        try:
            pref_raw = self._redis.get(self._pref_key(user_id)) or ""
            preferences_text = self._parse_preferences_text(pref_raw)
            summaries = self._get_top_summaries(user_id, top_k=5)
        except Exception as e:
            logger.warning(f"长期记忆读取失败，降级忽略：{e}")
            return ""

        semantic_summaries = []
        if query:
            try:
                query_embedding = self._embeddings.embed_query(query)
                semantic_summaries = milvus_client.search(user_id, query_embedding)
                self._increment_access(user_id, semantic_summaries)
            except Exception as e:
                logger.warning(f"语义记忆召回失败，降级忽略：{e}")

        semantic_summaries = [s for s in semantic_summaries if s not in summaries]

        parts = []
        if preferences_text:
            parts.append(f"用户偏好：{preferences_text}")
        if summaries:
            parts.append("最近会话摘要：\n" + "\n".join(f"- {s}" for s in summaries))
        if semantic_summaries:
            parts.append("语义相关历史摘要：\n" + "\n".join(f"- {s}" for s in semantic_summaries))
        return "\n".join(parts)

    def _parse_preferences_text(self, raw: str) -> str:
        """将结构化偏好 JSON 转为可读文本，兼容旧版纯字符串格式。"""
        if not raw:
            return ""
        try:
            data = json.loads(raw)
            facts = data.get("facts", [])
            if facts:
                return "；".join(
                    f["content"] for f in sorted(facts, key=lambda x: -x.get("confidence", 5))
                )
        except (json.JSONDecodeError, TypeError):
            return raw  # 旧版纯字符串，直接返回
        return ""

    def _get_top_summaries(self, user_id: str, top_k: int = 5) -> List[str]:
        """按综合重要性分数取前 K 条摘要。"""
        all_items = self._redis.zrange(self._sum_key(user_id), 0, -1, withscores=True)
        if not all_items:
            return []
        access_key = self._access_key(user_id)
        now = time.time()
        scored = []
        for member, ts in all_items:
            text = member if isinstance(member, str) else member.decode()
            count = int(self._redis.hget(access_key, _md5(text)) or 0)
            score = _importance_score(ts, count, now)
            scored.append((score, text))
        scored.sort(reverse=True)
        return [t for _, t in scored[:top_k]]

    def _increment_access(self, user_id: str, summaries: List[str]) -> None:
        """增加被语义召回的摘要的访问计数。"""
        if not summaries:
            return
        pipe = self._redis.pipeline()
        for s in summaries:
            pipe.hincrby(self._access_key(user_id), _md5(s), 1)
        pipe.expire(self._access_key(user_id), LTM_TTL_SECONDS)
        pipe.execute()

    # ── 写入 ──

    def save_session(self, user_id: str, messages: List[BaseMessage], llm: ChatOpenAI) -> None:
        """会话结束时，提取摘要和偏好，合并冲突后写入 Redis 和 Milvus。"""
        if not messages:
            return
        try:
            conversation = "\n".join(
                f"{'用户' if isinstance(m, HumanMessage) else 'AI'}：{m.content}"
                for m in messages
            )
            response = llm.invoke(_SUMMARIZE_PROMPT.format(conversation=conversation))
            result = self._parse_llm_json(response.content)

            pipe = self._redis.pipeline()

            if result.get("preferences"):
                merged = self._merge_preferences(user_id, result["preferences"], llm)
                pipe.set(self._pref_key(user_id), json.dumps(merged, ensure_ascii=False), ex=LTM_TTL_SECONDS)

            if result.get("summary"):
                self._add_summary_to_pipe(user_id, result["summary"], pipe)

            pipe.execute()

            if result.get("summary"):
                self._upsert_milvus(user_id, result["summary"], llm)

            logger.info(f"长期记忆已保存（含冲突消解）：user_id={user_id}")
        except Exception as e:
            logger.warning(f"长期记忆写入失败，降级忽略：{e}")

    def _merge_preferences(self, user_id: str, new_pref_text: str, llm: ChatOpenAI) -> dict:
        """用 LLM 将新偏好合并进现有结构化偏好列表，返回合并后的 dict。"""
        pref_raw = self._redis.get(self._pref_key(user_id)) or ""
        existing_facts = []
        if pref_raw:
            try:
                existing_facts = json.loads(pref_raw).get("facts", [])
            except (json.JSONDecodeError, TypeError):
                existing_facts = [{"content": pref_raw, "confidence": 5}]

        if not existing_facts:
            return {"facts": [{"content": new_pref_text, "confidence": 5}]}

        facts_str = "\n".join(
            f"- [{f.get('confidence', 5)}/10] {f['content']}" for f in existing_facts
        )
        response = llm.invoke(
            _MERGE_PREFERENCES_PROMPT.format(existing_facts=facts_str, new_preferences=new_pref_text)
        )
        merged = self._parse_llm_json(response.content)
        return merged if merged.get("facts") else {"facts": existing_facts}

    def _add_summary_to_pipe(self, user_id: str, summary: str, pipe) -> None:
        """将摘要写入 Redis Sorted Set，score = 当前时间戳。"""
        pipe.zadd(self._sum_key(user_id), {summary: time.time()})
        pipe.expire(self._sum_key(user_id), LTM_TTL_SECONDS)

        count = self._redis.zcard(self._sum_key(user_id))
        if count >= MAX_SUMMARIES:
            self._evict_lowest(user_id, keep=MAX_SUMMARIES - 1)

    def _evict_lowest(self, user_id: str, keep: int) -> None:
        """按综合重要性分数淘汰超限摘要，同步删除 Milvus 向量。"""
        all_items = self._redis.zrange(self._sum_key(user_id), 0, -1, withscores=True)
        if len(all_items) <= keep:
            return
        access_key = self._access_key(user_id)
        ids_key = self._ids_key(user_id)
        now = time.time()

        scored = []
        for member, ts in all_items:
            text = member if isinstance(member, str) else member.decode()
            count = int(self._redis.hget(access_key, _md5(text)) or 0)
            scored.append((_importance_score(ts, count, now), text))

        scored.sort()  # 升序，最低分在前
        to_remove = len(scored) - keep
        for _, text in scored[:to_remove]:
            h = _md5(text)
            # 删除 Redis 条目
            self._redis.zrem(self._sum_key(user_id), text)
            self._redis.hdel(access_key, h)
            # 同步删除 Milvus 向量
            milvus_id_raw = self._redis.hget(ids_key, h)
            if milvus_id_raw:
                milvus_client.delete([int(milvus_id_raw)])
                self._redis.hdel(ids_key, h)
            logger.debug(f"遗忘低重要性摘要：{text[:50]}…")

    def _upsert_milvus(self, user_id: str, summary: str, llm: ChatOpenAI) -> None:
        """写入 Milvus，插入前：① 检查近重复替换；② 用 LLM 检测偏好冲突并删除矛盾摘要。"""
        try:
            embedding = self._embeddings.embed_query(summary)

            # ① 近重复去重
            similar = milvus_client.find_similar(user_id, embedding)
            if similar:
                old_id, old_text = similar
                milvus_client.delete([old_id])
                self._redis.hdel(self._ids_key(user_id), _md5(old_text))
                logger.debug(f"替换近重复向量：{old_text[:40]}… → {summary[:40]}…")

            # ② 偏好冲突检测：召回语义相关摘要，让 LLM 判断矛盾
            related = milvus_client.find_related(user_id, embedding)
            if related:
                old_summaries_text = "\n".join(f"{i}. {text}" for i, (_, text) in enumerate(related))
                response = llm.invoke(
                    _CONFLICT_SUMMARY_PROMPT.format(
                        new_summary=summary,
                        old_summaries=old_summaries_text,
                    )
                )
                result = self._parse_llm_json(response.content)
                for idx in result.get("conflict_indices", []):
                    if idx < len(related):
                        conflict_id, conflict_text = related[idx]
                        milvus_client.delete([conflict_id])
                        self._redis.hdel(self._ids_key(user_id), _md5(conflict_text))
                        logger.debug(f"删除冲突摘要：{conflict_text[:40]}…")

            ids = milvus_client.insert(user_id, summary, embedding)
            if ids:
                self._redis.hset(self._ids_key(user_id), _md5(summary), ids[0])
                self._redis.expire(self._ids_key(user_id), LTM_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Milvus 向量写入失败，降级忽略：{e}")

    # ── 工具 ──

    @staticmethod
    def _parse_llm_json(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())

    def clear(self, user_id: str) -> None:
        """清除指定用户的全部长期记忆（Redis + Milvus）。"""
        try:
            # 先清 Milvus 向量
            ids_key = self._ids_key(user_id)
            # 按 TTL 清理用户所有过期前的向量（删全部用 0 时间戳）
            milvus_client.delete_expired(user_id, cutoff_timestamp=int(time.time()) + 1)
            # 再清 Redis
            self._redis.delete(
                self._pref_key(user_id),
                self._sum_key(user_id),
                self._access_key(user_id),
                ids_key,
            )
        except Exception as e:
            logger.warning(f"长期记忆清除失败：{e}")


long_term_memory_manager = LongTermMemoryManager()
