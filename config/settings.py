import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    # 豆包模型名称，填入火山引擎控制台的 Endpoint ID（格式：ep-xxxxxxxx-xxxxx）
    # 或直接使用模型名：doubao-pro-32k / doubao-pro-128k / doubao-lite-32k
    model: str = os.getenv("ARK_MODEL", "doubao-pro-32k")
    temperature: float = 0.1
    max_tokens: int = 2048
    api_key: str = os.getenv("ARK_API_KEY", "")
    base_url: str = os.getenv("ARK_BASE_URL", "https://ark.volces.com/api/v3")


@dataclass
class RAGServiceConfig:
    """RAG 微服务配置 —— 已有现成微服务，直接 HTTP 调用"""
    base_url: str = os.getenv("RAG_SERVICE_URL", "http://localhost:8080")
    endpoint: str = "/api/v1/query"
    timeout: int = 30
    api_key: str = os.getenv("RAG_SERVICE_API_KEY", "")


@dataclass
class MCPConfig:
    """MCP 工具服务端点配置"""
    weather_url: str = os.getenv("MCP_WEATHER_URL", "http://localhost:8081")
    map_url: str = os.getenv("MCP_MAP_URL", "http://localhost:8082")
    timeout: int = 10


llm_config = LLMConfig()
rag_service_config = RAGServiceConfig()
mcp_config = MCPConfig()
