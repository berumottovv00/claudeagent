"""
用户偏好查询工具
从长期记忆（Milvus）中检索用户历史偏好、常用地点、行为习惯。
"""
from langchain_core.tools import tool
from memory import session_memory_manager


@tool
def get_user_preference(session_id: str, category: str = "all") -> str:
    """获取用户历史偏好和常用信息。
    当需要为用户做个性化推荐、了解其出行习惯时使用。
    输入：session_id（用户会话ID），category（偏好类别：food/route/music/all）。
    输出：用户在该类别下的历史偏好描述。
    """
    # ----------------------------------------------------------------
    # 基础实现：从短期记忆中提取偏好信息（生产环境替换为 Milvus 长期记忆检索）
    #
    # 长期记忆检索示例（对接 Milvus 时替换）：
    # from pymilvus import connections, Collection
    # connections.connect(host=milvus_config.host, port=milvus_config.port)
    # collection = Collection("user_preferences")
    # query_vector = embed(f"{session_id} {category}")
    # results = collection.search(query_vector, "embedding", {"metric_type": "IP"}, limit=5)
    # return "\n".join([r.entity.get("preference") for r in results[0]])
    # ----------------------------------------------------------------

    history = session_memory_manager.get_history_messages(session_id)
    if not history:
        return f"暂无用户「{session_id}」的历史偏好记录。"

    # 从会话历史中简单提取偏好信息
    history_text = " | ".join([
        msg.content for msg in history if hasattr(msg, "content")
    ])
    return f"用户历史对话摘要（类别：{category}）：{history_text[:500]}"
