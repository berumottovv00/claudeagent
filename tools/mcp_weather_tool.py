"""
天气工具 —— 调用高德天气 API
"""
import os
from datetime import date
from typing import Any, Dict

import requests
from langchain_core.tools import tool

AMAP_MAPS_API_KEY = os.environ.get("AMAP_MAPS_API_KEY", "")


def _maps_weather(city: str, query_date: str) -> Dict[str, Any]:
    """调用高德天气 API，返回指定城市、日期的天气信息。"""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/weather/weatherInfo",
            params={
                "key": AMAP_MAPS_API_KEY,
                "city": city,
                "extensions": "all",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"error": f"获取天气失败：{data.get('info') or data.get('infocode')}"}

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return {"error": "暂无预报数据"}

        result = {"城市": forecasts[0]["city"]}
        for cast in forecasts[0]["casts"]:
            if cast["date"] != query_date:
                continue
            for key, name in {
                "dayweather": "天气",
                "daytemp": "温度",
                "daywind": "风向",
                "daypower": "风力",
            }.items():
                if cast.get(key):
                    result[name] = cast[key]

        if len(result) == 1:
            return {"error": f"未找到 {query_date} 的预报数据"}
        return result

    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败：{str(e)}"}


@tool
def get_weather(city: str, query_date: str = "") -> str:
    """查询指定城市的天气状况。
    当用户询问天气、路面状况、出行建议、雨雪天驾驶注意事项时使用此工具。
    输入：city（城市名或高德 adcode），query_date（日期 YYYY-MM-DD，默认今天）。
    输出：天气状况、温度、风向、风力等信息。
    """
    if not query_date:
        query_date = str(date.today())

    data = _maps_weather(city, query_date)

    if "error" in data:
        return data["error"]

    return "、".join(f"{k}：{v}" for k, v in data.items())