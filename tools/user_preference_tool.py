"""
用户偏好查询工具
从长期记忆（Redis + Milvus）中检索用户历史偏好、常用地点、行为习惯。
"""
from langchain_core.tools import tool
from memory import long_term_memory_manager


@tool
def get_user_preference(user_id: str, category: str = "all") -> str:
    """获取用户历史偏好和常用信息。
    当需要为用户做个性化推荐、了解其出行习惯时使用。
    输入：user_id（用户ID），category（偏好类别：food/route/music/all）。
    输出：用户在该类别下的历史偏好描述。
    """
    context = long_term_memory_manager.get_context(user_id, query=category)
    if not context:
        return f"暂无用户「{user_id}」的历史偏好记录。"
    return context