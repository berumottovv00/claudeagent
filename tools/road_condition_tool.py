"""
路况查询工具 —— 接口预留
查询指定路线或区域的实时路况、拥堵信息。
"""
from langchain_core.tools import tool


@tool
def get_road_condition(route: str) -> str:
    """查询指定路线的实时路况和拥堵情况。
    当需要了解某条路线是否拥堵、是否有事故或施工时使用。
    输入：路线描述（如"北京三环"、"从朝阳区到海淀区"）。
    输出：路况状态、拥堵路段、预计延误时间。
    """
    # ----------------------------------------------------------------
    # 接口预留：对接高德地图路况 API 时替换此处
    #
    # 真实调用示例：
    # resp = client.get(f"{mcp_config.map_url}/road_condition",
    #                   params={"route": route})
    # data = resp.json()
    # return f"路况：{data['status']}，拥堵路段：{data['congestion']}，延误：{data['delay']}"
    # ----------------------------------------------------------------

    return f"[路况接口预留] 路线：{route}，请对接实际路况服务后获取真实数据。"
