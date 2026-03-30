"""
多会话记忆管理
按 session_id 隔离，每个会话独立维护滑动窗口对话历史
"""
from collections import deque
from typing import Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class SessionMemoryManager:
    def __init__(self, window_k: int = 10):
        self.window_k = window_k
        # 每个 session 存储 (human, ai) 元组的滑动窗口
        self._sessions: Dict[str, deque] = {}

    def get_history_messages(self, session_id: str) -> List[BaseMessage]:
        """返回历史对话的 Message 对象列表，供 LangGraph agent 直接使用"""
        if session_id not in self._sessions:
            return []
        messages = []
        for human_msg, ai_msg in self._sessions[session_id]:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
        return messages

    def save_turn(self, session_id: str, human_msg: str, ai_msg: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.window_k)
        self._sessions[session_id].append((human_msg, ai_msg))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())


# 全局单例
session_memory_manager = SessionMemoryManager()
