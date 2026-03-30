"""
补能站查询工具 —— 接口预留
查询附近加油站或充电桩位置及可用状态。
"""
from langchain_core.tools import tool


@tool
def get_fuel_station(location: str, station_type: str = "all") -> str:
    """查询附近加油站或充电桩。
    当需要规划补能停靠点时使用，尤其是长途出行前。
    输入：location（当前位置或途经城市），station_type（gas=加油站 / charge=充电桩 / all=全部）。
    输出：最近补能站名称、距离、可用状态。
    """
    # ----------------------------------------------------------------
    # 接口预留：对接高德地图 POI 搜索 API 时替换此处
    #
    # 真实调用示例：
    # category = "加油站" if station_type == "gas" else "充电站"
    # resp = client.get(f"{mcp_config.map_url}/poi_search",
    #                   params={"location": location, "category": category, "radius": 5000})
    # ----------------------------------------------------------------

    return f"[补能站接口预留] 位置：{location}，类型：{station_type}，请对接实际地图服务后获取数据。"
