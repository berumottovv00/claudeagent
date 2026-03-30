"""
POI 兴趣点搜索工具 —— 接口预留
搜索附近餐厅、停车场、景点、商场等兴趣点。
"""
from langchain_core.tools import tool


@tool
def search_nearby_poi(location: str, category: str, radius_km: float = 5.0) -> str:
    """搜索附近兴趣点（餐厅、咖啡厅、景点、停车场等）。
    当需要为用户推荐附近地点时使用。
    输入：location（位置），category（类别，如"餐厅"、"咖啡厅"、"停车场"），radius_km（搜索半径，默认5km）。
    输出：周边匹配地点列表，含名称、距离、评分。
    """
    # ----------------------------------------------------------------
    # 接口预留：对接高德地图 / 百度地图 POI 搜索 API 时替换此处
    #
    # 真实调用示例：
    # resp = client.get(f"{mcp_config.map_url}/poi_search",
    #                   params={"location": location, "category": category,
    #                           "radius": int(radius_km * 1000)})
    # data = resp.json()
    # items = [f"{p['name']}（{p['distance']}m，评分{p['rating']}）" for p in data["pois"]]
    # return "\n".join(items)
    # ----------------------------------------------------------------

    return f"[POI搜索接口预留] 位置：{location}，类别：{category}，半径：{radius_km}km，请对接实际地图服务。"
