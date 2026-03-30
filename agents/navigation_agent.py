"""
导航规划 Sub Agent
综合路况、天气、补能需求，为用户制定最优出行方案。
内部运行独立的 ReAct 循环，不依赖 Orchestrator 的上下文。
"""
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import llm_config
from tools.mcp_weather_tool import get_weather
from tools.mcp_map_tool import get_navigation
from tools.road_condition_tool import get_road_condition
from tools.fuel_station_tool import get_fuel_station

NAVIGATION_SYSTEM_PROMPT = """你是专业的汽车导航规划助手，负责为用户制定完整的出行方案。

规划步骤（按需执行）：
1. 查询目标路线的实时路况，评估拥堵情况
2. 查询出发地或沿途天气，判断对驾驶的影响
3. 若行程较长或用户提到电量/油量不足，查询沿途补能站
4. 调用导航接口获取推荐路线和预计时长
5. 综合以上信息，输出包含路线建议、注意事项、补能提示的完整出行方案

输出要求：简洁实用，突出关键风险（如暴雨、严重拥堵）和行动建议。"""


class NavigationAgent:
    def __init__(self):
        llm = ChatOpenAI(
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )
        self._agent = create_react_agent(
            model=llm,
            tools=[get_road_condition, get_weather, get_navigation, get_fuel_station],
            prompt=NAVIGATION_SYSTEM_PROMPT,
        )

    def run(self, query: str) -> str:
        """执行导航规划，返回完整出行方案"""
        result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
        return result["messages"][-1].content


navigation_agent = NavigationAgent()
