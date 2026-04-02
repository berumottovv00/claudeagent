"""
用户偏好查询工具
从长期记忆（Redis + Milvus）中检索用户历史偏好、常用地点、行为习惯。
"""
from langchain_core.tools import tool
from memory import long_term_memory_manager


@tool
def get_user_preference(user_id: str, query: str = "") -> str:
    """获取用户历史偏好和常用信息。
    当需要为用户做个性化推荐、了解其出行习惯时使用。
    输入：user_id（用户ID），query（当前推荐场景的自然语言描述，用于召回语义相关的历史记忆，如"成都附近好吃的川菜馆"）。
    输出：用户的历史偏好及与当前场景相关的历史摘要。
    """
    context = long_term_memory_manager.get_context(user_id, query=query)
    if not context:
        return f"暂无用户「{user_id}」的历史偏好记录。"
    return context