"""
POI 兴趣点搜索工具 —— 对接高德地图周边搜索 API
搜索附近餐厅、停车场、景点、商场等兴趣点。
"""
import os
from typing import Any, Dict

import requests
from langchain_core.tools import tool

AMAP_MAPS_API_KEY = os.environ["AMAP_MAPS_API_KEY"]


def _around_search(location: str, radius: int, keywords: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/place/around",
            params={
                "key": AMAP_MAPS_API_KEY,
                "location": location,
                "radius": radius,
                "keywords": keywords,
            }
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"error": f"Around Search failed: {data.get('info') or data.get('infocode')}"}

        pois = [
            {
                "name": poi.get("name"),
                "address": poi.get("address"),
                "distance": poi.get("distance"),
            }
            for poi in data.get("pois", [])
        ]
        return {"pois": pois}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


@tool
def search_nearby_poi(location: str, category: str, radius_km: float = 5.0) -> str:
    """搜索附近兴趣点（餐厅、咖啡厅、景点、停车场等）。
    当需要为用户推荐附近地点时使用。
    输入：location（高德经纬度坐标，格式"经度,纬度"），category（类别，如"餐厅"、"咖啡厅"、"停车场"），radius_km（搜索半径，默认5km）。
    输出：周边匹配地点列表，含名称、地址、距离。
    """
    radius_m = int(radius_km * 1000)
    result = _around_search(location, radius_m, category)

    if "error" in result:
        return f"[POI搜索失败] {result['error']}"

    pois = result.get("pois", [])
    if not pois:
        return f"未找到附近 {category} 相关地点。"

    items = [
        f"{p['name']}（地址：{p['address'] or '未知'}，距离：{p['distance'] or '未知'}m）"
        for p in pois
    ]
    return "\n".join(items)