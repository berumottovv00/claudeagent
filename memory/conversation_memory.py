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

    def _key(self, user_id: str, session_id: str) -> str:
        return f"{user_id}:{session_id}"

    def get_history_messages(self, user_id: str, session_id: str) -> List[BaseMessage]:
        """返回历史对话的 Message 对象列表，供 LangGraph agent 直接使用"""
        key = self._key(user_id, session_id)
        if key not in self._sessions:
            return []
        messages = []
        for human_msg, ai_msg in self._sessions[key]:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
        return messages

    def save_turn(self, user_id: str, session_id: str, human_msg: str, ai_msg: str) -> None:
        key = self._key(user_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = deque(maxlen=self.window_k)
        self._sessions[key].append((human_msg, ai_msg))

    def clear(self, user_id: str, session_id: str) -> None:
        self._sessions.pop(self._key(user_id, session_id), None)

    def list_keys(self) -> List[str]:
        return list(self._sessions.keys())


# 全局单例
session_memory_manager = SessionMemoryManager()