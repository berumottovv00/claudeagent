"""
多会话记忆管理
按 user_id + session_id 隔离，每个会话独立维护滑动窗口对话历史
"""
from collections import deque
from typing import Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class SessionMemoryManager:
    def __init__(self, window_k: int = 10):
        self.window_k = window_k
        # key 格式："{user_id}:{session_id}"
        self._sessions: Dict[str, deque] = {}
        self._full_history: Dict[str, List] = {}

    def _key(self, user_id: str, session_id: str) -> str:
        return f"{user_id}:{session_id}"

    def get_history_messages(self, user_id: str, session_id: str) -> List[BaseMessage]:
        """返回滑动窗口内的历史消息，供 LangGraph agent 直接使用"""
        key = self._key(user_id, session_id)
        if key not in self._sessions:
            return []
        messages = []
        for human_msg, ai_msg in self._sessions[key]:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
        return messages

    def get_full_history(self, user_id: str, session_id: str) -> List[BaseMessage]:
        """返回完整历史消息（包含窗口外的轮次）"""
        key = self._key(user_id, session_id)
        messages = []
        for human_msg, ai_msg in self._full_history.get(key, []):
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
        return messages

    def save_turn(self, user_id: str, session_id: str, human_msg: str, ai_msg: str) -> None:
        key = self._key(user_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = deque(maxlen=self.window_k)
            self._full_history[key] = []
        self._sessions[key].append((human_msg, ai_msg))
        self._full_history[key].append((human_msg, ai_msg))

    def clear(self, user_id: str, session_id: str) -> None:
        key = self._key(user_id, session_id)
        self._sessions.pop(key, None)
        self._full_history.pop(key, None)

    def list_keys(self) -> List[str]:
        return list(self._sessions.keys())


# 全局单例
session_memory_manager = SessionMemoryManager()