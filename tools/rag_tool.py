"""
RAG 工具 —— 接口预留
调用已有的 RAG 微服务，检索汽车用户手册相关知识。
真实实现只需取消注释 HTTP 调用部分并填入配置即可。
"""
import httpx
from langchain_core.tools import tool
from config import rag_service_config


@tool
def rag_query(query: str) -> str:
    """查询汽车用户手册知识库。
    当用户询问车辆具体功能、操作方法、故障排查、技术规格、保养要求时使用此工具。
    输入：用户问题的核心关键词或完整问题。
    输出：从手册中检索到的相关知识片段。
    """
    # ----------------------------------------------------------------
    # 接口预留：对接 RAG 微服务时替换此处实现
    #
    # 真实调用示例：
    # try:
    #     with httpx.Client(timeout=rag_service_config.timeout) as client:
    #         resp = client.post(
    #             f"{rag_service_config.base_url}{rag_service_config.endpoint}",
    #             json={"query": query, "top_k": 5},
    #             headers={"Authorization": f"Bearer {rag_service_config.api_key}"},
    #         )
    #         resp.raise_for_status()
    #         data = resp.json()
    #         contexts = "\n---\n".join(data["contexts"])
    #         return f"手册相关内容：\n{contexts}"
    # except httpx.TimeoutException:
    #     return "RAG 服务超时，无法获取手册信息"
    # except Exception as e:
    #     return f"RAG 服务异常：{str(e)}"
    # ----------------------------------------------------------------

    return f"[RAG 接口预留] 查询：「{query}」，请对接实际 RAG 微服务后获取真实手册内容。"
