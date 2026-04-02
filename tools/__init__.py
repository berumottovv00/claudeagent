from langchain_core.tools import tool

from .rag_tool import rag_query
from .mcp_weather_tool import get_weather
from .mcp_map_tool import get_navigation


@tool
def navigate(query: str) -> str:
    """导航规划工具：处理复杂出行规划需求，综合考虑实时路况、天气、补能站等多项因素。
    当用户需要路线规划、出行准备、长途驾驶建议时使用此工具。
    输入：用户的出行需求描述（含出发地、目的地、特殊需求等）。
    输出：完整的出行方案，含路线建议、注意事项、补能提示。
    """
    # 懒加载，避免与 navigation_agent 循环导入
    from agents.navigation_agent import navigation_agent
    return navigation_agent.run(query)


@tool
def recommend(query: str, user_id: str = "default") -> str:
    """个性化推荐工具：根据用户偏好和当前情境（位置、时间、天气）推荐附近地点或活动。
    当用户询问"去哪吃饭"、"附近有什么好玩的"、"推荐一下"等个性化推荐需求时使用。
    输入：用户的推荐需求描述（含位置、偏好、场景等信息），以及 user_id（用于读取历史偏好）。
    输出：2-3 个个性化推荐，附带推荐理由。
    """
    from agents.recommendation_agent import recommendation_agent
    full_query = f"{query}\n[user_id: {user_id}]"
    return recommendation_agent.run(full_query)


# Orchestrator 可用的全量工具列表
ALL_TOOLS = [rag_query, get_weather, get_navigation, navigate, recommend]
