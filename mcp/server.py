"""
MCP Server 示例
暴露两个工具：POI 周边搜索 + 天气查询。

MCP 协议基于 JSON-RPC 2.0，通信方式支持：
  - stdio（本地子进程）
  - HTTP + SSE（远程服务）

启动方式：
  python mcp/server.py
"""
import asyncio
import json
import os

import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ----------------------------------------------------------------
# 创建 MCP Server 实例
# ----------------------------------------------------------------
server = Server("demo-server")


# ----------------------------------------------------------------
# 第一步：声明工具列表
# 客户端调用 tools/list 时，Server 返回这里定义的工具描述。
# 大模型根据工具描述决定调用哪个工具、传什么参数。
# ----------------------------------------------------------------
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_nearby_poi",
            description="搜索附近兴趣点（餐厅、咖啡厅、景点、停车场等）。输入经纬度坐标和类别，返回周边匹配地点列表。",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "高德经纬度坐标，格式：'经度,纬度'，例如 '104.065735,30.659462'",
                    },
                    "category": {
                        "type": "string",
                        "description": "搜索类别，例如：餐厅、咖啡厅、停车场",
                    },
                    "radius_km": {
                        "type": "number",
                        "description": "搜索半径（千米），默认 5.0",
                        "default": 5.0,
                    },
                },
                "required": ["location", "category"],
            },
        ),
        Tool(
            name="get_weather",
            description="查询指定城市的实时天气。输入城市名称，返回天气描述。",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：成都、北京",
                    }
                },
                "required": ["city"],
            },
        ),
    ]


# ----------------------------------------------------------------
# 第二步：实现工具逻辑
# 客户端调用 tools/call 时，Server 在这里执行真实逻辑并返回结果。
# ----------------------------------------------------------------
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_nearby_poi":
        result = _search_poi(
            location=arguments["location"],
            category=arguments["category"],
            radius_km=arguments.get("radius_km", 5.0),
        )
        return [TextContent(type="text", text=result)]

    if name == "get_weather":
        result = _get_weather(city=arguments["city"])
        return [TextContent(type="text", text=result)]

    return [TextContent(type="text", text=f"未知工具：{name}")]


# ----------------------------------------------------------------
# 工具实现
# ----------------------------------------------------------------
def _search_poi(location: str, category: str, radius_km: float) -> str:
    api_key = os.environ.get("AMAP_MAPS_API_KEY", "")
    try:
        resp = requests.get(
            "https://restapi.amap.com/v3/place/around",
            params={
                "key": api_key,
                "location": location,
                "radius": int(radius_km * 1000),
                "keywords": category,
            },
            timeout=10,
        )
        data = resp.json()
        if data["status"] != "1":
            return f"搜索失败：{data.get('info')}"
        pois = data.get("pois", [])[:5]
        if not pois:
            return f"未找到附近 {category}。"
        return "\n".join(
            f"{p['name']}（{p.get('address', '地址未知')}，距离 {p.get('distance', '?')}m）"
            for p in pois
        )
    except Exception as e:
        return f"请求失败：{e}"


def _get_weather(city: str) -> str:
    api_key = os.environ.get("AMAP_MAPS_API_KEY", "")
    try:
        resp = requests.get(
            "https://restapi.amap.com/v3/weather/weatherInfo",
            params={"key": api_key, "city": city, "extensions": "base"},
            timeout=10,
        )
        data = resp.json()
        if data["status"] != "1" or not data.get("lives"):
            return f"天气查询失败：{data.get('info')}"
        live = data["lives"][0]
        return (
            f"{live['city']}：{live['weather']}，"
            f"气温 {live['temperature']}℃，"
            f"风向 {live['winddirection']}，"
            f"风力 {live['windpower']} 级"
        )
    except Exception as e:
        return f"请求失败：{e}"


# ----------------------------------------------------------------
# 启动 Server（stdio 模式）
# ----------------------------------------------------------------
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())