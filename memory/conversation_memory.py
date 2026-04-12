"""
多会话记忆管理
按 user_id + session_id 隔离，每个会话用一个 list 存全量历史。
get_history_messages 只返回最近 window_k 轮用于推理，全量用于摘要和 flush。
字符数超过阈值时在后台线程异步触发摘要写入，不阻塞当前请求。
"""
import os
import threading
from typing import Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# 粗估：中文约 1.5 字/token，默认 3000 token 上限 ≈ 4500 字符
_MAX_CHARS = int(os.environ.get("SESSION_MAX_CHARS", str(int(3000 * 1.5))))


class SessionMemoryManager:
    def __init__(self, window_k: int = 10, max_chars: int = _MAX_CHARS):
        self.window_k = window_k
        self.max_chars = max_chars
        # key 格式："{user_id}:{session_id}"，value 为完整对话轮次列表
        self._sessions: Dict[str, List] = {}
        # 各 session 当前累计字符数
        self._char_counts: Dict[str, int] = {}
        # 增量 flush 回调，由 OrchestratorAgent 注入
        self._flush_callback: Optional[Callable[[str, str], None]] = None

    def _key(self, user_id: str, session_id: str) -> str:
        return f"{user_id}:{session_id}"

    def _to_messages(self, turns: list) -> List[BaseMessage]:
        messages = []
        for human_msg, ai_msg in turns:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
        return messages

    def get_history_messages(self, user_id: str, session_id: str) -> List[BaseMessage]:
        """返回最近 window_k 轮历史消息，供 LangGraph agent 推理使用。"""
        key = self._key(user_id, session_id)
        return self._to_messages(self._sessions.get(key, [])[-self.window_k:])

    def get_full_history(self, user_id: str, session_id: str) -> List[BaseMessage]:
        """返回完整历史消息。"""
        key = self._key(user_id, session_id)
        return self._to_messages(self._sessions.get(key, []))

    def get_and_clear_oldest(self, user_id: str, session_id: str, n: int) -> List[BaseMessage]:
        """取出并删除最旧的 n 轮，返回 BaseMessage 列表（供增量摘要使用）。"""
        key = self._key(user_id, session_id)
        history = self._sessions.get(key, [])
        oldest, self._sessions[key] = history[:n], history[n:]
        # 同步扣减已清除轮次的字符数
        removed_chars = sum(len(h) + len(a) for h, a in oldest)
        self._char_counts[key] = max(0, self._char_counts.get(key, 0) - removed_chars)
        return self._to_messages(oldest)

    def save_turn(self, user_id: str, session_id: str, human_msg: str, ai_msg: str) -> None:
        key = self._key(user_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = []
            self._char_counts[key] = 0
        self._sessions[key].append((human_msg, ai_msg))
        self._char_counts[key] += len(human_msg) + len(ai_msg)
        # 字符数超限时在后台线程异步触发摘要 flush
        if self._flush_callback and self._char_counts[key] >= self.max_chars:
            threading.Thread(
                target=self._flush_callback,
                args=(user_id, session_id),
                daemon=True,
            ).start()

    def clear(self, user_id: str, session_id: str) -> None:
        key = self._key(user_id, session_id)
        self._sessions.pop(key, None)
        self._char_counts.pop(key, None)

    def list_keys(self) -> List[str]:
        return list(self._sessions.keys())


# 全局单例
session_memory_manager = SessionMemoryManager()
