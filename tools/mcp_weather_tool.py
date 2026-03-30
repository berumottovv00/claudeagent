"""
MCP 天气工具 —— 接口预留
通过 MCP 协议调用天气服务，获取实时天气和路面状况。
真实实现对接高德天气 / 和风天气 API 时替换此处即可。
"""
import httpx
from langchain_core.tools import tool
from config import mcp_config


@tool
def get_weather(city: str, date: str = "today") -> str:
    """查询指定城市的天气状况。
    当用户询问天气、路面状况、出行建议、雨雪天驾驶注意事项时使用此工具。
    输入：city（城市名），date（日期，默认 today）。
    输出：天气状况、温度、路面状态等信息。
    """
    # ----------------------------------------------------------------
    # 接口预留：对接 MCP 天气服务时替换此处实现
    #
    # 真实调用示例（对接和风天气 API）：
    # try:
    #     with httpx.Client(timeout=mcp_config.timeout) as client:
    #         resp = client.get(
    #             f"{mcp_config.weather_url}/weather",
    #             params={"city": city, "date": date},
    #         )
    #         resp.raise_for_status()
    #         data = resp.json()
    #         return (
    #             f"城市：{city}\n"
    #             f"天气：{data['weather']}\n"
    #             f"温度：{data['temp']}°C\n"
    #             f"路面状态：{data['road_condition']}"
    #         )
    # except httpx.TimeoutException:
    #     return "天气服务超时，请稍后重试"
    # except Exception as e:
    #     return f"天气服务异常：{str(e)}"
    # ----------------------------------------------------------------

    return f"[MCP 天气接口预留] 城市：{city}，日期：{date}。请对接实际天气服务后获取真实数据。"
