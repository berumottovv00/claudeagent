"""
个性化推荐 Sub Agent
结合用户历史偏好、当前时间/位置、实时天气，提供贴合情境的个性化推荐。
内部运行独立的 ReAct 循环，不依赖 Orchestrator 的上下文。
"""
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import llm_config
from tools.user_preference_tool import get_user_preference
from tools.poi_search_tool import search_nearby_poi
from tools.mcp_weather_tool import get_weather

RECOMMENDATION_SYSTEM_PROMPT = """你是智能汽车的个性化推荐助手，基于用户偏好和当前情境提供贴心建议。

推荐步骤（按需执行）：
1. 获取用户历史偏好，了解其口味、习惯、常去场所
2. 查询当前天气，判断天气对活动的影响（如雨天不适合户外）
3. 搜索附近符合条件的兴趣点（餐厅、咖啡厅、停车场等）
4. 结合偏好 + 天气 + 位置，筛选出最匹配的 2-3 个推荐

输出要求：
- 每个推荐说明理由（为什么适合该用户）
- 语气自然亲切，像朋友建议而非机械列表
- 若偏好数据不足，主动询问用户偏好"""


class RecommendationAgent:
    def __init__(self):
        llm = ChatOpenAI(
            model=llm_config.model,
            temperature=0.3,  # 推荐场景适当提高创造性
            max_tokens=llm_config.max_tokens,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )
        self._agent = create_react_agent(
            model=llm,
            tools=[get_user_preference, search_nearby_poi, get_weather],
            prompt=RECOMMENDATION_SYSTEM_PROMPT,
        )

    def run(self, query: str) -> str:
        """执行个性化推荐，返回推荐结果"""
        result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
        return result["messages"][-1].content


recommendation_agent = RecommendationAgent()
