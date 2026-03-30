"""
MCP 地图导航工具 —— 接口预留
通过 MCP 协议调用地图服务，获取导航路线和路况信息。
真实实现对接高德地图 API 时替换此处即可。
"""
import httpx
from langchain_core.tools import tool
from config import mcp_config


@tool
def get_navigation(origin: str, destination: str, mode: str = "driving") -> str:
    """查询从出发地到目的地的导航路线。
    当用户询问路线规划、驾车时长、路况、距离时使用此工具。
    输入：origin（出发地），destination（目的地），mode（出行方式：driving/walking/transit，默认 driving）。
    输出：推荐路线、距离、预计行驶时长。
    """
    # ----------------------------------------------------------------
    # 接口预留：对接 MCP 地图服务时替换此处实现
    #
    # 真实调用示例（对接高德地图 API）：
    # try:
    #     with httpx.Client(timeout=mcp_config.timeout) as client:
    #         resp = client.get(
    #             f"{mcp_config.map_url}/navigation",
    #             params={"origin": origin, "destination": destination, "mode": mode},
    #         )
    #         resp.raise_for_status()
    #         data = resp.json()
    #         return (
    #             f"出发地：{origin} → 目的地：{destination}\n"
    #             f"距离：{data['distance']}\n"
    #             f"预计时间：{data['duration']}\n"
    #             f"推荐路线：{data['route_summary']}"
    #         )
    # except httpx.TimeoutException:
    #     return "地图服务超时，请稍后重试"
    # except Exception as e:
    #     return f"地图服务异常：{str(e)}"
    # ----------------------------------------------------------------

    return f"[MCP 地图接口预留] 路线：{origin} → {destination}，方式：{mode}。请对接实际地图服务后获取真实数据。"
